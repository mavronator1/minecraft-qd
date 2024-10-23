"""
Scoring fns for the Flat Genotype. 
"""
import jax
import jax.numpy as jnp
import jax.random as jrand
import qdax.core.containers.mapelites_repertoire as me_repertoire


from typing import Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from flat.genotype import FlatGenotype

from common.scoring_fn import ScoringFn
from jax import lax, Array
from functools import partial
from jax.tree_util import tree_map
from common.jaxqueue import JaxList, JaxQueue

from structure.scoring_fns import Custom
from gameplay.scoring_fns import StructureDistributionScore
from hierarchy.scoring_fns import MaximizePath


class FlatScore(ScoringFn):

    def __init__(self, G:'type[FlatGenotype]'):

        self.G = G
        self.s_G = G.s_G
        self.g_G = G.g_G


class CombinedScore(FlatScore):

    """
    Takes elements of the scoring functions of each layer in the hierachy into one scoring function

    """

    def __init__(self, 
                 
                 G,

                 min_size:int,

                 min_distance:int,
                 gameplay_range:list,

                 g_weight:float,
                 d_weight:float,
                 cc_weight:float,
                 bonus_weight:float,

                 enemy_val:int=2,
                 trap_val:int=3,
                 treasure_val:int=4,
        
                 ):

        super().__init__(G)

        self.min_size = min_size
        self.min_distance = min_distance
        self.gameplay_range = tuple(gameplay_range)

        self.g_weight = g_weight
        self.d_weight = d_weight
        self.cc_weight = cc_weight
        self.bonus_weight = bonus_weight

        self.enemy_val = enemy_val
        self.trap_val = trap_val
        self.treasure_val = treasure_val

    @partial(jax.jit, static_argnames=("self",))
    def score(self, genotype: Array, RNGKey: Array) -> Tuple[float, Array]:

        structure_genotype, _ = genotype

        chambers, connections, (start, end) = structure_genotype

        # chambers connected
        chamber_idx = jnp.argwhere(~jnp.all(chambers == 0, axis=(1,2)), size=self.s_G.max_chambers, fill_value=-1)
        connected = check_connected(connections, chamber_idx)

        # size penalty
        structure_genotype = tree_map(lambda x: jnp.expand_dims(x, 0), structure_genotype)
        structure = self.s_G.express(structure_genotype)[0]
        size = jnp.sum(structure)
        size_penalty = (jnp.maximum(self.min_size - size, 0) / self.min_size) 
        
        # structure_penalty:
        structure_penalty = ((1-connected) + size_penalty) / 2

        # chamber connection fitness
        chamber_size = jnp.sum(chambers)
        connection_size = jnp.sum(self.s_G.generate_connections(connections, chambers))
        chamber_ratio = chamber_size / (chamber_size + connection_size)

        cc_fitness = self.pw_chamber_connection_fn(chamber_ratio)
        
        # build level
        genotype = tree_map(lambda x: jnp.expand_dims(x, 0), genotype)
        level = self.G.express(genotype)[0]

        start = jnp.argwhere(level==8, size=1)[0]
        end =  jnp.argwhere(level==9, size=1)[0]
        level_size = jnp.sum(level >= 1) # playable area

        traps = jnp.sum(level == self.trap_val) 
        enemies = jnp.sum(level == self.enemy_val) 
        treasure = jnp.sum(level == self.treasure_val) 

        total_g = (traps+enemies+treasure) 

        g = total_g / level_size
        g_fitness = self.pw_gameplay_fn(g)

        d = is_traversable_bfs_jit(level, start, end)
        d_fitness = self.pw_distance_fn(d)

        # bonus fitness: is treasure reachable? 

        bonus = self.treasure_bonus(level, start, end)

        fitness = ((self.g_weight*g_fitness + self.d_weight*d_fitness + self.bonus_weight*bonus) / 
                   (self.g_weight+self.d_weight+self.bonus_weight))
        
        fitness = jnp.maximum(fitness, 0)
        
        # bds:

        start += 1 # 1 to 20
        end += 1 # 1 to 20 

        traps_enemies_sum = (total_g - treasure)

        traps_ratio = jnp.where(traps_enemies_sum == 0, 
                                jnp.array([0]), 
                                jnp.round(jnp.array([traps / traps_enemies_sum]), 1))
        
        enemies_ratio = jnp.where(traps_enemies_sum == 0, 
                                  jnp.array([0]), 
                                  jnp.round(jnp.array([enemies / traps_enemies_sum]), 1))

        bd = jnp.concatenate([start/20, end/20, traps_ratio, enemies_ratio])

        return fitness, bd


    @partial(jax.jit, static_argnames=("self",))
    def treasure_bonus(self, level, start, end):
        # only look for n treasures
        treasures = jnp.argwhere(level==self.treasure_val, size=3, fill_value=-1)
        alt_level = level.at[end].set(1) # remove end
        search_fn = partial(is_traversable_bfs_jit, start=start)

        def treasure_search(treasure_coord, level):

            pred = jnp.all(treasure_coord == -1)
            r, c = treasure_coord[0], treasure_coord[1]
            return lax.cond(pred, lambda x: -1.0, lambda x: search_fn(level=level.at[r, c].set(9), end=x), treasure_coord)
    
        reachable_treasure = jax.vmap(treasure_search, in_axes=(0, None))(treasures, alt_level)
        treasure_score = jnp.sum(reachable_treasure >= 1) / jnp.sum(reachable_treasure != -1) 

        return jnp.where(jnp.isnan(treasure_score), 0, treasure_score)
    
    @partial(jax.jit, static_argnames=("self",))
    def pw_distance_fn(self, x):
        """
        Piecewise function for distance
        
        """

        cond_list = [
            (x < self.min_distance),
            (x >= self.min_distance)
        ]

        m = 1/self.min_distance

        func_list = [
            lambda x: x*m,
            1,
            0
        ]

        return jnp.piecewise(x, cond_list, func_list)
    
    @partial(jax.jit, static_argnames=("self",))
    def pw_gameplay_fn(self, x):
        """
        Piecewise function for gameplay
        
        """

        g0, g1 = self.gameplay_range

        cond_list = [
            ((x >= 0) & (x < g0)),
            ((x >= g0) & (x <= g1)),
            ((x > g1) & (x <= 1))
        ]

        m1 = 1/g0
        c3 = 1 + g1/g0        
    
        func_list = [
            lambda x: m1*x ,
            1,
            lambda x: -m1*x + c3,
            0
        ]

        return jnp.maximum(jnp.piecewise(x, cond_list, func_list), 0)
    
    @partial(jax.jit, static_argnames=("self",))
    def pw_chamber_connection_fn(self, x):

        """
        Piecewise function for gameplay
        
        """

        min_ratio = 0.3
        max_ratio = 0.5

        cond_list = [

            ((x >= 0) & (x < min_ratio)),
            ((x >= min_ratio) & (x <= max_ratio)),
            ((x > max_ratio) & (x <= 1))
        ]

        m1 = 1/min_ratio
        c3 = 1 + max_ratio/min_ratio        
    
        func_list = [
            lambda x: m1*x ,
            1,
            lambda x: -m1*x + c3,
            0
        ]

        return jnp.maximum(jnp.piecewise(x, cond_list, func_list), 0)

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
def is_traversable_bfs_jit(level:Array, start:Array, end:Array) -> Array:

    Q = JaxQueue(500, 3, jnp.int16, batch_size=8)
    L = JaxList(500, 2, batch_size=8)

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
        pos = pos_d[0:2].astype(jnp.int32)
        distance = pos_d[2]

        found = jnp.all(pos == end)

        neighbors = get_neighbors(pos)

        # elimante invalid corners
        corners = neighbors[4:]

        corner_check = jnp.array([

                [~jnp.logical_and(jnp.isin(level[pos[0]+1, pos[1]], jnp.array([3, 4])), 
                                    jnp.isin(level[pos[0], pos[1]+1], jnp.array([3, 4])))],
                [~jnp.logical_and(jnp.isin(level[pos[0]-1, pos[1]], jnp.array([3, 4])), 
                                    jnp.isin(level[pos[0], pos[1]+1], jnp.array([3, 4])))],
                [~jnp.logical_and(jnp.isin(level[pos[0]+1, pos[1]], jnp.array([3, 4])), 
                                    jnp.isin(level[pos[0], pos[1]-1], jnp.array([3, 4])))],
                [~jnp.logical_and(jnp.isin(level[pos[0]-1, pos[1]], jnp.array([3, 4])), 
                                    jnp.isin(level[pos[0], pos[1]-1], jnp.array([3, 4])))]
                                    ])
        
        corners = jnp.where(corner_check, corners, jnp.array([-1, -1]))        
        neighbors = neighbors.at[4:].set(corners)
        
        def vmap_fn(neighbor, distance):

            row, col = neighbor[0].astype(int), neighbor[1].astype(int)

            pred = jnp.logical_and(
                ~L.is_in(visited, neighbor),
                jnp.logical_not(jnp.isin(level[row, col], jnp.array([0, 3, 4])))
            )
            
            return lax.select(pred,
                              jnp.concatenate((neighbor, jnp.array([distance+1]))),
                              jnp.array([jnp.inf, jnp.inf, jnp.inf]))
        
        neighbor_batch = jax.vmap(vmap_fn, in_axes=(0, None))(neighbors, distance)
        queue, q_order = Q.batch_insert(queue, q_order, neighbor_batch)
        visited, v_size = L.batch_append(visited, v_size, neighbor_batch[:, :2])

        return queue, q_order, visited, v_size, found, distance
    
    _, _, _, _, found, distance = lax.while_loop(cond_fn, body_fn, (queue, q_order, visited, v_size, False, 0))
    
    return found*distance


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
        