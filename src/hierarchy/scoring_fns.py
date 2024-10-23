"""
Scoring fns for the Hierarchy Genotype

"""
import jax
import jax.numpy as jnp
import jax.random as jrand
import qdax.core.containers.mapelites_repertoire as me_repertoire


from typing import Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from hierarchy.genotype import MiddleLayer

from common.scoring_fn import ScoringFn
from jax import lax, Array
from functools import partial
from jax.tree_util import tree_map
from common.jaxqueue import JaxList, JaxQueue


class HierarchyScore(ScoringFn):

    def __init__(self, G:'type[MiddleLayer]'):
        
        self.G = G
        self.s_G = G.s_G
        self.g_G = G.g_G

class MaximizePath(HierarchyScore):
    
    """
    Aims to maximize the distance from start to finish 

    """

    def __init__(self, 
                 
                 G:'type[MiddleLayer]', 
                 fn_type:str,
        
                 min_distance:int,
                 gameplay_range:list,
                 
                 d_weight:float,
                 g_weight:float,
                 bonus_weight:float,

                 n_treasure_search:int = 3,
            
                 enemy_val:int=2, 
                 trap_val:int=3, 
                 treasure_val:int=4):
        
            super().__init__(G)

            self.min_distance = min_distance # fitness is 1 after this value
            self.gameplay_range = tuple(gameplay_range) # 1 between these values

            self.d_weight = d_weight # distance fn weight
            self.g_weight = g_weight # gameplay fn weight
            self.bonus_weight = bonus_weight

            self.n_treasure_search = n_treasure_search

            self.enemy_val = enemy_val
            self.trap_val = trap_val
            self.treasure_val = treasure_val


    @partial(jax.jit, static_argnames=("self",))
    def score(self, genotype: Array, RNGKey: Array) -> Tuple[float, Array]:

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
        
        # bds:

        start += 1 # 1 to 20
        end += 1 # 1 to 20 

        traps_enemies_sum = (total_g - treasure)

        traps_ratio = jnp.where(traps_enemies_sum == 0, 
                                jnp.array([0]), 
                                jnp.array([traps / traps_enemies_sum]))
        
        enemies_ratio = jnp.where(traps_enemies_sum == 0, 
                                  jnp.array([0]), 
                                  jnp.array([enemies / traps_enemies_sum]))


        bd = jnp.concatenate([start/20, end/20, enemies_ratio])

        return fitness, bd
    
    @partial(jax.jit, static_argnames=("self",))
    def treasure_bonus(self, level, start, end):
        # only look for n treasures
        treasures = jnp.argwhere(level==self.treasure_val, size=self.n_treasure_search, fill_value=-1)
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

    
class PathFindScore(HierarchyScore):
    """

    Scores generated levels using a BFS to determine if it is navigatble. 
    
    """

    def __init__(self, 
                 
                 G:'type[MiddleLayer]', 

                 enemy_norm:int, # max enemies before norm
                 trap_norm:int, # 
                 treasure_norm:int,
                 size_norm:float,
                 feature_range:Tuple,
                 min_distance:int,

                 enemy_val:int=2, 
                 trap_val:int=3, 
                 treasure_val:int=4):
        

            super().__init__(G)

            self.enemy_norm = enemy_norm
            self.trap_norm = trap_norm
            self.treasure_norm = treasure_norm
            self.size_norm = size_norm
            self.feature_range = feature_range
            self.min_distance = min_distance

            self.enemy_val = enemy_val
            self.trap_val = trap_val
            self.treasure_val = treasure_val

    @partial(jax.jit, static_argnames=("self",))
    def score(self, genotype: Array, RNGKey: Array) -> Tuple[float, Array]:

        structure_bd, _ = genotype

        # get the nearest structure bd
        structure_cell_idx = me_repertoire.get_cells_indices(structure_bd, self.G.s_filled_centroids)
        bd = tree_map(lambda x: lax.collapse(x[structure_cell_idx], 0, 2), self.G.s_repertoire.descriptors)
        connection_score = bd[0]

        # build level
        genotype = tree_map(lambda x: jnp.expand_dims(x, 0), genotype)
        level = self.G.express(genotype)[0]

        # number of element in level
        n_traps = jnp.sum(level == self.trap_val)
        n_enemies = jnp.sum(level == self.enemy_val)
        n_treasure = jnp.sum(level == self.treasure_val)
        n_playable = jnp.sum(level >= 1) # playable area

        # fitness
        start = jnp.argwhere(level==8, size=1)[0]
        end =  jnp.argwhere(level==9, size=1)[0]

        fitness, distance = self.fitness(n_playable, 
                                         n_traps, 
                                         n_enemies, 
                                         n_treasure,
                                         level,
                                         start,
                                         end)
        
        # BDs
        normalized_size = jnp.minimum(jnp.sum(level >= 1) / self.size_norm, 1)
        mazeness = self.maze_score(n_traps, connection_score)
        dungeoness = self.dungeon_score(n_enemies, connection_score)
    
        return fitness, jnp.array([mazeness, dungeoness, normalized_size])

    @partial(jax.jit, static_argnames=("self",))
    def maze_score(self, n_traps: float, connection_score: float) -> Array:
        normalized_traps = jnp.minimum((n_traps / self.trap_norm), 1)
        return ((normalized_traps + connection_score) / 2)
        
    @partial(jax.jit, static_argnames=("self",))
    def dungeon_score(self, n_enemies: Array, connection_score: float) -> Array:
        normalized_enemies = jnp.minimum((n_enemies / self.enemy_norm), 1)
        return ((normalized_enemies + (1 - connection_score))/ 2)
    
    @partial(jax.jit, static_argnames=("self",))
    def fitness(self, n_playable:int, n_traps:int, n_enemies:int, n_treasure:int, level:Array, start:Array, end:Array) -> float:

        target_feature_min = self.feature_range[0]
        target_feature_max = self.feature_range[1]

        feature_ratio = (n_traps+n_enemies+n_treasure) / n_playable

        conditions = [

            (0 < feature_ratio) & (feature_ratio < target_feature_min),
            (target_feature_min <= feature_ratio) & (feature_ratio <= target_feature_max),
            (target_feature_max < feature_ratio ) & (feature_ratio < 1)
        ]

        fns = [

            lambda x: x,
            1,
            lambda x: -x+1,
            0
        ]

        feature_penalty = jnp.piecewise(feature_ratio, conditions, fns)

        distance = is_traversable_bfs_jit(level, start, end) 
        traversability_penalty = jnp.maximum(self.min_distance - distance, 0)  / self.min_distance

        total_penalty = (traversability_penalty + feature_penalty) / 2

        fitness = 1 - total_penalty

        return fitness, distance

class GameTypeScore(HierarchyScore):

    """
    Scores the generated levels based on the 'type' of level they create:

    Levels:

        Maze -> traps, treasure, winding structure
        Dungeon -> enemies, treasure, chamber structure 
    
    """


    def __init__(self, 
                 
                 G:'type[MiddleLayer]', 

                 enemy_norm:int, # max enemies before norm
                 trap_norm:int, # 
                 treasure_norm:int,
                 size_norm:float,
                 feature_range:Tuple,

                 enemy_val:int=2, 
                 trap_val:int=3, 
                 treasure_val:int=4):
        

        super().__init__(G)

        self.enemy_norm = enemy_norm
        self.trap_norm = trap_norm
        self.treasure_norm = treasure_norm
        self.size_norm = size_norm
        self.feature_range = feature_range

        self.enemy_val = enemy_val
        self.trap_val = trap_val
        self.treasure_val = treasure_val

    @partial(jax.jit, static_argnames=("self",))
    def score(self, genotype: Array, RNGKey: Array) -> Tuple[float, Array]:

        structure_bd, _ = genotype

        # get the nearest structure bd
        structure_cell_idx = me_repertoire.get_cells_indices(structure_bd, self.G.s_filled_centroids)
        bd = tree_map(lambda x: lax.collapse(x[structure_cell_idx], 0, 2), self.G.s_repertoire.descriptors)
        connection_score = bd[0]

        # build level
        genotype = tree_map(lambda x: jnp.expand_dims(x, 0), genotype)
        level = self.G.express(genotype)

        # number of element in level
        n_traps = jnp.sum(level == self.trap_val)
        n_enemies = jnp.sum(level == self.enemy_val)
        n_treasure = jnp.sum(level == self.treasure_val)
        n_floor = jnp.sum(level == 1)

        # BDs
        normalized_size = jnp.minimum(jnp.sum(level >= 1) / self.size_norm, 1)
        mazeness = self.maze_score(n_traps, connection_score)
        dungeoness = self.dungeon_score(n_enemies, connection_score)

        fitness = self.fitness(n_floor, n_traps, n_enemies, n_treasure)
        
        return fitness, jnp.array([mazeness, dungeoness, normalized_size])

    @partial(jax.jit, static_argnames=("self",))
    def maze_score(self, n_traps: float, connection_score: float) -> Array:
        normalized_traps = jnp.minimum((n_traps / self.trap_norm), 1)
        return ((normalized_traps + connection_score) / 2)
        
    @partial(jax.jit, static_argnames=("self",))
    def dungeon_score(self, n_enemies: Array, connection_score: float) -> Array:
        normalized_enemies = jnp.minimum((n_enemies / self.enemy_norm), 1)
        return ((normalized_enemies + (1 - connection_score))/ 2)
    
    @partial(jax.jit, static_argnames=("self",))
    def fitness(self, n_floor:int, n_traps:int, n_enemies:int, n_treasure:int) -> float:
        
        feature_ratio = (n_traps+n_enemies+n_treasure) / n_floor

        conditions = [
            (0 < feature_ratio) & (feature_ratio < self.feature_range[0]),
            (self.feature_range[0] <= feature_ratio) & (feature_ratio <= self.feature_range[1]),
            (self.feature_range[1] < feature_ratio ) & (feature_ratio < 1)
        ]

        fns = [
            lambda x: x/self.feature_range[0],
            1,
            lambda x: (1-x) / (1-self.feature_range[1]),
            0
        ]

        return jnp.piecewise(feature_ratio, conditions, fns)
    
class BasicScore(HierarchyScore):

    """
    BD only and no fitness
    
    """

    def __init__(self, G:'type[MiddleLayer]', max_difficulty:int):

        super().__init__(G)
        self.max_difficulty = max_difficulty


    @partial(jax.jit, static_argnames=("self",))
    def score(self, genotype: Array, RNGKey: Array) -> Tuple[float, Array]:

        
        s_bd, g_bd = genotype
        structure_type = self.get_structure_type(s_bd)

        # need to re-batch to build
        genotype = tree_map(lambda x: jnp.expand_dims(x, 0), genotype)
        level = self.G.express(genotype)

        size = jnp.sum(level >= 1) / (self.s_G.size**2)
        # level_type = self.get_gameplay_type(level)
        difficulty = self.get_difficulty(level)

        bds = jnp.array([difficulty, structure_type, size])

        return 1.0, bds
    

    @partial(jax.jit, static_argnames=("self",))
    def get_structure_type(self, s_bd:Array) -> float:
        # gets nearest idx
        structure_cell_idx = me_repertoire.get_cells_indices(s_bd, self.G.s_filled_centroids)
        # extract from desciptors
        bds = tree_map(lambda x: lax.collapse(x[structure_cell_idx], 0, 2), self.G.s_repertoire.descriptors)
        # return the first bd (ratio of connections to chambers)
        return bds[0]


    @partial(jax.jit, static_argnames=("self",))
    def get_difficulty(self, genotype:Array) -> float:
    
        difficulty_sum = jnp.sum(genotype==2) + jnp.sum(genotype==3)
        return jnp.minimum(difficulty_sum, self.max_difficulty) / self.max_difficulty
    
    
    @partial(jax.jit, static_argnames=("self",))
    def get_gameplay_type(self, genotype:Array):
        """
        Assuming 3 gameplay types

        """
        _, counts = jnp.unique(genotype, return_counts=True, size=5)
        gameplay_elements = jnp.sum(counts[2:])

        guard_room = counts[2] + counts[4] # monsters + treasure
        dungeon_room  = counts[3] + counts[4] # traps + treasue
        danger_room = counts[3] + counts [4] # monsters + traps

        return jnp.array([guard_room, dungeon_room, danger_room]) / gameplay_elements

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
        
        

        





 




