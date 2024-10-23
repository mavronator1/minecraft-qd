"""

Structure Genotype for MAP-Elites

"""


import jax 
import jax.numpy as jnp
import jax.random as jrand
import cv2
import numpy as np
import itertools
import math


from jax import Array, lax
from jax import tree_util
from typing import Tuple
from functools import partial
from common.genotype import Genotype
from logging import Logger
from structure.scoring_fns import StructureScore


class StructureGenotype(Genotype):

  
    def __init__(self, 
                 
                 batch_size:int,

                 structure_size:int=20, 
                 max_chambers:int =5,
                 max_connections:int = 20,

                 expansion_threshold:float = 0.5,
                 expansion_perimeter:int = 1,
                
                 mutation_chamber_p:float = 0.3,
                 mutation_add_chamber_p:float = 0.1,
                 mutation_connection_p:float = 0.1,
                 mutation_start_end_p:float = 0.05,

                 cross_over_p:float = 0.2,
                 logger:Logger=None,
                 
                 ):

        if max_connections < math.comb(max_chambers, 2):
         raise ValueError("max_connections must cover at least max_chambers C 2")
        
        self.logger = logger

        if self.logger:
            self.logger.info("Structure Genotype")

        self.size = structure_size # length/width of strucutre 
        self.fill_val = 0 # non structure value
        self.start_val = 8 # indicating start 
        self.end_val = 9 # indiciating end
 
        self.max_chambers = max_chambers # max number of chambers in a structure
        self.max_connections = max_connections # max number of connection structures 
        self.max_chamber_connections = math.comb(max_chambers, 2)

        self.expansion_threshold = expansion_threshold # p of expanding from around structure
        self.expansion_perimeter = expansion_perimeter # number of steps away from structure to possibly expand 

        self.mutation_chamber_p = mutation_chamber_p # p of mutating chambers 
        self.mutation_add_chamber_p = mutation_add_chamber_p # p of adding new chamber in mutation
        self.mutation_connection_p = mutation_connection_p # p of adding new mutation
        self.mutation_start_end_p = mutation_start_end_p # p of changing start and end 
        self.cross_over_p = cross_over_p # p of cross over

        self.X_COORDS, self.Y_COORDS = jnp.meshgrid(jnp.arange(self.size), 
                                                    jnp.arange(self.size))
        
        self.indicies = lax.collapse(jnp.indices((20, 20)), 1, 3).transpose(1,0)
        
        self.batch_size = batch_size

        self.scoring_fn = None
        
    @partial(jax.jit, static_argnames=("self",))
    def generate_genotypes(self, RNGKey) -> Tuple[Tuple[Array, Array], Array]:

        """
        Generates a batch of structure genotype
        
        """

        def batch_generation(key):

            # GENERATE CHAMBERS

            keys = jrand.split(key, num=self.max_chambers)
            chambers, keys = jax.vmap(self.generate_chamber)(keys)

            # randomly mask some
            key, subkey = jrand.split(keys[0])
            chamber_mask = jrand.uniform(key, (self.max_chambers, 1, 1)) < 0.4 # keep 40% 
            chambers = jnp.where(chamber_mask, chambers, jnp.full_like(chambers[0],0))

            # GENERATE CONNECTIONS

            combinations = self.get_valid_chamber_combinations(chambers)
            # randomly choose connection points
            key, subkey = jrand.split(key)
            combinations_mask = jrand.uniform(subkey, (self.max_chamber_connections, 1)) > 0.5
            connections = jnp.where(combinations_mask, combinations, 
                                    jnp.full_like(combinations[0], -1))
    
            # GENERATE START AND END POINTS
            key, subkey = jrand.split(key)
            start, key = self.generate_point(chambers, subkey)
            key, subkey = jrand.split(key)
            end, key = self.generate_point(chambers, subkey)

            start_end = (start, end)

            return (chambers, connections, start_end)
        
        key, subkey = jrand.split(RNGKey)
        keys = jrand.split(subkey, self.batch_size)
        (chambers, connections, start_end) = jax.vmap(batch_generation)(keys)
        
        return (chambers, connections, start_end), key
    
    def set_scoring_fn(self, scoring_fn:StructureScore, *scoring_fn_args, **scoring_fn_kwargs):

        self.scoring_fn = scoring_fn(self, *scoring_fn_args, **scoring_fn_kwargs)
        if self.logger:
            self.logger.info('Set scoring fn!')

    @partial(jax.jit, static_argnames=("self",))
    def score_genotypes(self, genotype: Array, RNGKey: Array) -> Tuple[float, Array, None, Array]:
        """
        Scoring function for the structure genotype 

        BD: 1. coverage (fraction of size)
            2. centroid of entire structure
            3. second order moments 
            
        """

        key, subkey = jrand.split(RNGKey)
        keys = jrand.split(subkey, self.batch_size)
        fitnesses, descriptor = jax.vmap(self.scoring_fn)(genotype, keys)
        
        return fitnesses, descriptor, None, key

    @partial(jax.jit, static_argnames=("self",))
    def variation_fn(self, x1:Array, x2:Array, RNGKey:Array):

        return super().variation_fn(x1, x2, RNGKey)

    
    @partial(jax.jit, static_argnames=("self",))
    def mutation_fn(self, x1, RNGKey):

        """

        Mutate by:
         # 1. expanding the chambers size and adding new chambers
         # 2. changing the connections mask by adding a pivot point. 
         # 3. changing start/end location 
    
        """ 

        def batch_mutation(x1, RNGKey):

            chambers, connections, start_end = x1
            valid_chambers_mask = ~jnp.all(chambers==0, axis=(1, 2))
            
            # CHAMBER MUTATION:

            # randomly mutate the current chambers 

            key, subkey = jrand.split(RNGKey)
            chamber_mutation_mask = ((jrand.uniform(subkey, (self.max_chambers,)) < self.mutation_chamber_p) 
                                    & valid_chambers_mask)
            
            def true_fn1(X):
                c, k = self.expand_chamber(*X)
                return c, k
            
            def false_fn1(X):
                c, k = X
                return c, k

            cond_fn = lambda p, c, k: jax.lax.cond(p, true_fn1, false_fn1, (c, k))        
            keys = jrand.split(key, num=self.max_chambers)
            chambers, keys = jax.vmap(cond_fn, in_axes=0)(chamber_mutation_mask, chambers, keys)

            # randomly add new chambers 

            key, subkey = jrand.split(keys[0])
            chamber_addition_mask = ((jrand.uniform(subkey, (self.max_chambers, )) < self.mutation_add_chamber_p) 
                                    & ~valid_chambers_mask)
            
            def true_fn2(X):
                c, k = self.generate_chamber(X[1])
                return c, k
            
            def false_fn2(X):
                c, k = X
                return c, k
            
            cond_fn = lambda p, c, k: jax.lax.cond(p, true_fn2, false_fn2, (c,k))
            keys = jrand.split(key, num=self.max_chambers)
            chambers, keys = jax.vmap(cond_fn, in_axes=0)(chamber_addition_mask, chambers, keys)
            
            # CONNECTION MUTATION

            # introduce new connections or remove them

            combinations = self.get_valid_chamber_combinations(chambers)
            connections_mask = jnp.all(combinations == connections, axis=1)[:, jnp.newaxis]

            key, subkey = jrand.split(keys[0])
            connection_ps = (jrand.uniform(subkey, (self.max_chamber_connections, 1)) 
                            <  self.mutation_connection_p)
                    
            connections_mask = jnp.where(connection_ps,
                                        ~connections_mask,
                                        connections_mask)

            connections = jnp.where(connections_mask, combinations, 
                                    jnp.full_like(combinations[0], -1))
            
            # START END MUTATION
            start_point, end_point = start_end

            key, subkey = jrand.split(key)
            pred = jrand.uniform(subkey, (2, )) < self.mutation_start_end_p
            key, subkey = jrand.split(key)

            start_mutated = jnp.where(
                pred[0],
                self.generate_point(chambers, subkey)[0],
                start_point
            )

            key, subkey = jrand.split(key)

            end_mutated = jnp.where(
                pred[1],
                self.generate_point(chambers, subkey)[0],
                end_point
            )

            start_end = (start_mutated, end_mutated)

            return chambers, connections, start_end
    
        key, subkey = jrand.split(RNGKey)
        keys = jrand.split(subkey, self.batch_size)
        y = jax.vmap(batch_mutation, in_axes=0)(x1, keys)

        return y, key
             
    @partial(jax.jit, static_argnames=("self",))
    def express(self, x:Tuple[Array, Array, Array]) -> Array:

        """
        Assembles chambers and connections from genotype
        to complete structure 
        
        """
        
        def batch_express(x):
 
            chambers, connections, (start, end) = x
            chamber_structure = jnp.any(chambers, axis=0) * 1
            connection_structure = self.generate_connections(connections, chambers)

            structure = chamber_structure | connection_structure

            start_coord = tuple(start)[1:]
            end_coord = tuple(end)[1:]

            structure = structure.at[start_coord].set(8)
            structure = structure.at[end_coord].set(9)

            return structure
        
        return jax.vmap(batch_express)(x)

    @partial(jax.jit, static_argnames=("self",))
    def express_no_start_end(self, x:Tuple[Array, Array, Array]) -> Tuple[Array, Tuple]:
        """
        Expresses but does not apply start, end, returing that seperately
        """

        chambers, connections, (start, end) = x
        

        def batch_express(x):
            
            chambers, connections = x

            chamber_structure = jnp.any(chambers, axis=0) * 1
            connection_structure = self.generate_connections(connections, chambers)

            structure = chamber_structure | connection_structure

            return structure
        
        return jax.vmap(batch_express)((chambers, connections)), (start, end)
    
    @partial(jax.jit, static_argnames=("self",))
    def apply_start_end(self, start_end_batch, level_batch) -> Array:

        """
        Helper function to set start and end values for a batch of levels
        
        """
        def apply_start_end(start_end, level):

            start, end = start_end

            start_coord = tuple(start)[1:]
            end_coord = tuple(end)[1:]

            level = level.at[start_coord].set(8)
            level = level.at[end_coord].set(9)

            return level

        return jax.vmap(apply_start_end)(start_end_batch, level_batch)

    @partial(jax.jit, static_argnames=("self",))
    def generate_chamber(self, RNGKey) -> Tuple[Array, Array]:

        """
        Generate a chamber from a random point in the structure
        """

        canvas = jnp.full((self.size, self.size), 0)

        key, subkey = jrand.split(RNGKey)

        x, y = jrand.randint(subkey, shape=(2,), minval=0, maxval=self.size-1)
                
        canvas = canvas.at[(y, x)].set(1)
        canvas, key = self.expand_chamber(canvas, key)

        return canvas, key

    @partial(jax.jit, static_argnames=("self",))
    def expand_chamber(self, canvas:Array, RNGKey:Array) -> Tuple[Array, Array]:

        """
        Randomly expands a chamber from existing points on the canvas
        """

        chamber_coords = jnp.argwhere(canvas != self.fill_val, size=self.size**2, fill_value=jnp.nan)

        def get_distance(coord):
            return lax.select(jnp.all(jnp.isnan(coord)), 
                              jnp.full((self.size, self.size), jnp.inf),
                              jnp.maximum(abs(coord[0] - self.Y_COORDS), abs(coord[1] - self.X_COORDS)))
        
        distances = jax.vmap(get_distance, in_axes=0)(chamber_coords)
        distance_mask = jnp.min(distances, axis=0)

        key, subkey = jrand.split(RNGKey)
        random_mask = jrand.normal(subkey, (self.size, self.size)) 
        random_mask = jnp.where(distance_mask > self.expansion_perimeter, jnp.nan, random_mask)
        
        expansion_mask = jnp.where(random_mask < self.expansion_threshold, 
                              jnp.full((self.size, self.size), 1),
                              jnp.full((self.size, self.size), 0))
        
        canvas = expansion_mask | canvas

        return canvas, key

    @partial(jax.jit, static_argnames=("self",))
    def generate_connections(self, connections:Array, chambers:Array) -> Array:

        """
        Generates the connection structure

        """

        def connect(connection):
            
            # get centroid for each chamber
            chamber0, chamber1 = chambers[connection[0]], chambers[connection[1]]
            c0, c1 = self.calculate_centroid(chamber0), self.calculate_centroid(chamber1)

            # form the connection structure
            connection_canvas = jnp.full((self.size, self.size), 0)
            connection_points = jnp.linspace(c0, c1, num=20, dtype=jnp.int8)

            # do not set any points if all connection points are same (therefore no connection)
            return lax.select(jnp.all(connection_points[0] == connection_points),
                              connection_canvas,
                              connection_canvas.at[connection_points[:, 0], connection_points[:, 1]].set(1))
        
        connection_structure = jax.vmap(connect)(connections)
        connection_structure = jnp.any(connection_structure, axis=0)*1

        return connection_structure
    
    @partial(jax.jit, static_argnames=("self",))
    def generate_point(self, chambers:Array, RNGKey:Array) -> Tuple[Array, Array]:

        """
        Chooses where to place a 'point' on a random valid chamber at a random valid location, 
        returning the point location
        
        """

        all_chamber_dist = ~jnp.all(chambers==0, axis=(1,2)) 

        key, subkey = jrand.split(RNGKey)
        chamber = jrand.choice(subkey, jnp.arange(0, self.max_chambers), (1,), p=all_chamber_dist)[0]

        chamber_dist = lax.collapse(chambers[chamber]==1, 0, 2)
        key, subkey = jrand.split(key)
        position = jrand.choice(subkey, self.indicies, (1,), p=chamber_dist)[0]

        # return point as array
        point = jnp.reshape(jnp.array([chamber, position[0], position[1]]), (3,))

        return point, key

    @partial(jax.jit, static_argnames=("self",))                    
    def get_valid_chamber_combinations(self, chambers:Array):

        chamber_ids = jnp.arange(0, self.max_chambers)
        combinations = jnp.array(list(itertools.combinations(chamber_ids, 2)))

        # remove combinations that do have a generated chamber
        
        empty_chambers = jnp.all(chambers==0, axis=(1,2))
        combinations_mask = empty_chambers[combinations[:,0]] | empty_chambers[combinations[:,1]]
        combinations = jnp.where(~combinations_mask[:, jnp.newaxis],
                                 combinations,
                                 jnp.full_like(combinations[0], -1))
        return combinations

    @partial(jax.jit, static_argnames=("self",))
    def calculate_centroid(self, canvas:Array) -> Array:

        """
        Get centroid from a canvas

        """
        m00 = jnp.sum(canvas)

        m10 = jnp.sum(self.X_COORDS * canvas)
        m01 = jnp.sum(self.Y_COORDS * canvas)
        
        centroid = jnp.array([m01/m00, m10/m00]) 
    
        return centroid
    

# debugging
if __name__ == "__main__":
    G = StructureGenotype(5)
    g = G.generate_genotypes(jrand.PRNGKey(0))

