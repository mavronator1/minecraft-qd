"""
Scoring fns for the Structure Genotype

"""
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jrand

from common.jaxqueue import JaxList, JaxQueue
from typing import Tuple, TYPE_CHECKING
from collections import deque


if TYPE_CHECKING:
    from structure.genotype import StructureGenotype

from common.scoring_fn import ScoringFn
from jax import lax, Array
from itertools import product
from functools import partial
from jax.tree_util import tree_map



class StructureScore(ScoringFn):

    def __init__(self, G:'type[StructureGenotype]'):

        self.s_G = G 

class CVScore(StructureScore):
    """
    Uses techniques from CV such as image centroids and
    second order moments to calculate the BD
    
    """

    @partial(jax.jit, static_argnames=("self",))
    def score(self, genotype: Array, RNGKey: Array) -> Tuple[float, Array]:

        # need to re-batch to build
        genotype = tree_map(lambda x: jnp.expand_dims(x, 0), genotype)
        structure = self.s_G.express(genotype)

        # calculate overall centroid
        centroid = calculate_centroid(structure, self.s_G.X_COORDS, self.s_G.Y_COORDS)
        
        # calculate second order moments 
        mu02, mu20 = calculate_2nd_moments(centroid, structure, self.s_G.X_COORDS, self.s_G.Y_COORDS)

        # calculate fullness
        fullness = jnp.sum(structure) / self.s_G.size ** 2
    
        descriptor = jnp.array([centroid[0], centroid[1], mu02, mu20, fullness]) 

        return 1.0, descriptor
    

class Custom(StructureScore):

    """ 
    work in progress..

    """
    def __init__(self, G:'type[StructureGenotype]', min_size:int, min_distance:int, min_chambers:int):
        super().__init__(G)

        self.min_size = min_size
        self.min_distance = min_distance # min distance between start and end
        self.min_chambers = min_chambers

    @partial(jax.jit, static_argnames=("self",))
    def score(self, genotype: Array, RNGKey: Array) -> Tuple[float, Array]:

        chambers, connections, (start, end) = genotype

        genotype = tree_map(lambda x: jnp.expand_dims(x, 0), genotype)
        structure = self.s_G.express(genotype)[0]
        
        # no longer need chamber id once structure is built
        start_pos = start[1:]
        end_pos = end[1:]

        # fitness:

        # chambers connected
        chamber_idx = jnp.argwhere(~jnp.all(chambers == 0, axis=(1,2)), size=self.s_G.max_chambers, fill_value=-1)
        connected = check_connected(connections, chamber_idx)

        # traversable 
        distance = is_traversable_bfs_jit(structure, start_pos, end_pos)
        traversable_penalty = jnp.maximum(self.min_distance - distance, 0)  / self.min_distance

        # size penalty
        size = jnp.sum(structure == 1)
        size_penalty = (jnp.maximum(self.min_size - size, 0) / self.min_size) 


        overall_penalty = ((1-connected) + traversable_penalty + size_penalty) / 3

        fitness = jnp.maximum(1 - overall_penalty, 0)

        # BDs

        chamber_size = jnp.sum(chambers) / size
        connection_size = jnp.sum(self.s_G.generate_connections(connections, chambers)) / size

        chamber_ratio = jnp.where((chamber_size + connection_size)==0, 
                                  0, 
                                  chamber_size / (chamber_size + connection_size))

        # num chambers 
        chamber_num = jnp.array([jnp.sum(chamber_idx != -1)]) / 5

        start_pos = jnp.maximum(start_pos, 0) / 20
        end_pos = jnp.maximum(end_pos, 0) / 20


        bd = jnp.concatenate([start_pos, end_pos, jnp.array([chamber_ratio]), chamber_num])

        return fitness, bd
        
@jax.jit
def is_traversable_bfs_jit(structure:Array, start:Array, end:Array) -> Array:

    Q = JaxQueue(500, 3, jnp.int16, 8)
    L = JaxList(500, 2, 8)

    queue, q_order = Q.new_queue()
    visited, v_size = L.new_list()

    distance = 0
    
    queue, q_order = Q.insert(queue, q_order, jnp.hstack((start, distance)))
    visited, v_size = L.append(visited, v_size, start)

    def cond_fn(x):
        queue, q_order, visited, v_size, found, distance = x

        return jnp.logical_and(~found, Q.len(queue) > 0)

    def body_fn(x):

        queue, q_order, visited, v_size, found, distance = x

        queue, pos_d = Q.pop(queue)
        pos = pos_d[0:2]
        distance = pos_d[2]

        found = jnp.all(pos == end)

        neighbors = get_neighbors(pos)
        
        def vmap_fn(neighbor, distance):

            row, col = neighbor[0].astype(int), neighbor[1].astype(int)

            pred = jnp.logical_and(~L.is_in(visited, neighbor), 
                                   jnp.not_equal(structure[row, col], 0))
            
            return lax.select(pred,
                              jnp.concatenate((neighbor, jnp.array([distance+1]))),
                              jnp.array([jnp.inf, jnp.inf, jnp.inf]))
        
        neighbor_batch = jax.vmap(vmap_fn, in_axes=(0, None))(neighbors, distance)
        queue, q_order = Q.batch_insert(queue, q_order, neighbor_batch)
        visited, v_size = L.batch_append(visited, v_size, neighbor_batch[:, :2])

        return queue, q_order, visited, v_size, found, distance
    
    _, _, _, _, found, distance = lax.while_loop(cond_fn, body_fn, (queue, q_order, visited, v_size, False, 0))
    

    return found * distance



@jax.jit
def get_neighbors(pos):

    down = jnp.where(pos[0]+1 < 20, jnp.array([pos[0]+1, pos[1]]), jnp.array([-1, -1]))
    up = jnp.where(pos[0]-1 > -1, jnp.array([pos[0]-1, pos[1]]), jnp.array([-1, -1]))
    right = jnp.where(pos[1]+1 < 20, jnp.array([pos[0], pos[1]+1]), jnp.array([-1, -1]))
    left = jnp.where(pos[1]-1 > -1, jnp.array([pos[0], pos[1]-1]), jnp.array([-1, -1]))
    down_right = jnp.where(jnp.logical_and(pos[0]+1 < 20,  pos[1]+1 < 20),  jnp.array([pos[0]+1, pos[1]+1]), jnp.array([-1, -1]))
    up_right = jnp.where(jnp.logical_and(pos[0]-1 > -1,  pos[1]+1 < 20),  jnp.array([pos[0]-1, pos[1]+1]), jnp.array([-1, -1]))
    down_left = jnp.where(jnp.logical_and(pos[0]+1 < 20,  pos[1]-1 > -1),  jnp.array([pos[0]+1, pos[1]-1]), jnp.array([-1, -1]))
    up_left = jnp.where(jnp.logical_and(pos[0]-1 > -1,  pos[1]-1 > -1),  jnp.array([pos[0]-1, pos[1]-1]), jnp.array([-1, -1]))
    
    return jnp.array([
        down,
        up,
        right,
        left,
        down_right,
        up_right,
        down_left,
        up_left
    ])

@jax.jit
def check_connected(connections, chamber_idxes):

    graph = jnp.full((5, 10), jnp.inf)
    graph_nodes_n = jnp.full((5, 1), 0)

    # scan fn to add nodes to graph:
    def scan_f(carry, x):

        pred = ~jnp.any(x == -1)

        def true_fn(carry, x):

            graph, graph_nodes_n = carry

            graph = graph.at[x[0], graph_nodes_n[x[0]]].set(x[1])
            graph = graph.at[x[1], graph_nodes_n[x[1]]].set(x[0])
            graph_nodes_n = graph_nodes_n.at[x[0]].set(graph_nodes_n[x[0]]+1)
            graph_nodes_n = graph_nodes_n.at[x[1]].set(graph_nodes_n[x[1]]+1)
            
            return graph, graph_nodes_n
            
        def false_fn(carry, x):
            return carry
        
        (graph, graph_nodes_n) = lax.cond(pred, true_fn, false_fn, carry, x)
        return (graph, graph_nodes_n), x

    (graph, graph_nodes_n), _ = lax.scan(scan_f, (graph, graph_nodes_n), connections)

    num_chambers = jnp.sum(chamber_idxes != -1)
    node = chamber_idxes[0]

    S = JaxQueue(50, 1, batch_size=10, stack=True) 
    stack, stack_order = S.new_queue()
    stack, stack_order = S.insert(stack, stack_order, node)

    V = JaxList(50, 1, batch_size=10)
    visited, visited_size = V.new_list()

    def cond_fn(X):

        stack, stack_order, visited, visited_size = X

        return S.len(stack) > 0 

    def body_fn(X):

        stack, stack_order, visited, visited_size = X

        stack, row = S.pop(stack)
        node = jnp.array([row[0]], dtype=jnp.int32) # convert to array for is_in

        def true_fn(x):
            
            stack, stack_order, visited, visited_size = x

            visited, visited_size = V.append(visited, visited_size, node)
            neighbors = graph[node[0]] # remove from array for index

            def vmap_fn(neighbor):
                pred = V.is_in(visited, jnp.array([neighbor])) # convert to array for is_in
                return jnp.where(pred, jnp.inf, neighbor)

            neighbor_batch = jax.vmap(vmap_fn)(neighbors)
            neighbor_batch = jnp.expand_dims(neighbor_batch, 1)
            stack, stack_order = S.batch_insert(stack, stack_order, neighbor_batch)

            y = stack, stack_order, visited, visited_size
            return y
    
        def false_fn(x):
            return x
        
        pred = ~V.is_in(visited, node)

        stack, stack_order, visited, visited_size = lax.cond(pred, true_fn, false_fn, (stack, stack_order, visited, visited_size))
        Y = stack, stack_order, visited, visited_size 

        return Y

    _, _, _, visited_size = lax.while_loop(cond_fn, body_fn, (stack, stack_order, visited, visited_size))

    return visited_size == num_chambers

@jax.jit
def calculate_centroid(canvas:Array, x_coords:Array, y_coords:Array) -> Tuple[float, float]:

    """
    Get centroid from a canvas
    
    Returns x_centroid, y_centroid

    """

    m00 = jnp.sum(canvas)

    m10 = jnp.sum(x_coords * canvas)
    m01 = jnp.sum(y_coords * canvas)

    x_centroid = m10/m00
    y_centroid = m01/m00
    
    return x_centroid, y_centroid

@partial(jax.jit, static_argnames=("x_coords","y_coords"))
def calculate_2nd_moments(centroid:Array, canvas:Array, x_coords:Array, y_coords:Array):
    """
    Calculate 2nd order central moments 

    """
    y_hat, x_hat = centroid[0], centroid[1]

    mu20 = jnp.sum((x_coords - x_hat)**2 * canvas)
    mu02 = jnp.sum((y_coords - y_hat)**2 * canvas)

    return mu02, mu20







