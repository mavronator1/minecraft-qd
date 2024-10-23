import jax 
import jax.numpy as jnp
import jax.random as jrand 
import jax.nn as jnn
import scipy.stats as stats
import qdax.core.containers.mapelites_repertoire as me_repertoire
import os

from hydra.utils import get_original_cwd
from hydra.core.global_hydra import GlobalHydra

from jax import vmap, lax, Array, tree_util
from functools import partial
from typing import Tuple, List
from itertools import product
from common.genotype import Genotype
from functools import partial
from logging import Logger

from structure.genotype import StructureGenotype
from gameplay.genotype import GameplayGenotype
# from flat.scoring_fns import FlatScore 
from hierarchy.scoring_fns import MaximizePath



class FlatGenotype(Genotype):

    """
    Entire hierarchy flattened into one! 

    """

    def __init__(self,
                
                batch_size:int,
                structure_args:dict,
                gameplay_args:dict,

                logger:Logger=None
                ):
        
        self.s_G = StructureGenotype(batch_size = batch_size,
                                     logger=logger,
                                     **structure_args)
        self.g_G = GameplayGenotype(batch_size=batch_size,
                                    logger=logger,
                                    **gameplay_args)
        
        self.batch_size = batch_size
        self.logger = logger
        self.scoring_fn = None


    def generate_genotypes(self, RNGKey: Array) -> Tuple[Array, Array]:

        structure_genotype, key = self.s_G.generate_genotypes(RNGKey)
        gameplay_genotype, key = self.g_G.generate_genotypes(key)
        genotype = (structure_genotype, gameplay_genotype)
        
        return genotype, key
    
    def set_scoring_fn(self, scoring_fn:MaximizePath, *scoring_fn_args, **scoring_fn_kwargs) -> None:

        self.scoring_fn = scoring_fn(self, *scoring_fn_args, **scoring_fn_kwargs)
        if self.logger:
            self.logger.info('Set scoring fn!')

    def score_genotypes(self, genotype: Array, RNGKey: Array) -> Tuple[float | Array | None]:
        
        assert self.scoring_fn is not None, 'scoring fn not set!'
        
        key, subkey = jrand.split(RNGKey)
        keys = jrand.split(subkey, self.batch_size)
        fitness, bds = jax.vmap(self.scoring_fn)(genotype, keys)

        return fitness, bds, None, key
    
    def mutation_fn(self, x1, RNGKey) -> Tuple[Array | Array]:

        return super().mutation_fn(x1, RNGKey)

    def variation_fn(self, x1, x2, RNGKey) -> Tuple[Array | Array]:

        """
        Sends x1 structure to mutation and crosses over x1 and x2 gamelpay
        
        """
        structure_genotype_1, gameplay_genotype_1 = x1
        _, gameplay_genotype_2 = x2

        mutated_structure, key = self.s_G.mutation_fn(structure_genotype_1, RNGKey)
        mutated_gameplay, key = self.g_G.variation_fn(gameplay_genotype_1, 
                                                      gameplay_genotype_2,
                                                      key)
        
        mutated_genotype = mutated_structure, mutated_gameplay

        return mutated_genotype, key

    def express(self, x) -> Array:

        structure_genotypes, gameplay_genotypes = x

        structure_batch, start_end_batch = self.s_G.express_no_start_end(structure_genotypes)    
        level_batch = jax.vmap(self.g_G.express, in_axes=0)(gameplay_genotypes, structure_batch) * structure_batch
        level_batch = self.s_G.apply_start_end(start_end_batch, level_batch)

        return level_batch
