"""
Genotype for layer above others in hierarchy

"""


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

from qdax.core.emitters.mutation_operators import isoline_variation
from gameplay.genotype import GameplayGenotype
from structure.genotype import StructureGenotype
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from hierarchy.scoring_fns import HierarchyScore 


class MiddleLayer(Genotype):

    """
    Middle layer in hierarchy, composing genotypes
    from the structure and gameplay repertoires. 

    """

    def __init__(self, 
                 
                 s_repertoire_path:str,
                 s_config_path:str,
                 g_repertoire_path:str,
                 g_config_path:str,

                 iso_sigma:float,
                 line_sigma:float,

                 batch_size:int, # batch size to use

                 logger: Logger = None

                ):
        
        from common.utils import load # this avoids circular import that I need to fix later...
        

        if GlobalHydra().is_initialized():
            s_repertoire_path = os.path.join(get_original_cwd(), s_repertoire_path)
            s_config_path = os.path.join(get_original_cwd(), s_config_path)
            g_repertoire_path = os.path.join(get_original_cwd(), g_repertoire_path)
            g_config_path = os.path.join(get_original_cwd(), g_config_path)

        self.s_repertoire, self.s_G = load(s_repertoire_path, s_config_path)
        self.g_repertoire, self.g_G = load(g_repertoire_path, g_config_path)

        self.s_centroids = (
            self.s_repertoire.centroids.at[jnp.any(jnp.isnan(self.s_repertoire.descriptors), axis=1)].set(jnp.inf)
        )
        self.s_genotypes = self.s_repertoire.genotypes

        gameplay_mask = jnp.round(jnp.sum(self.g_repertoire.descriptors, axis=1), decimals=1) == 1.0
        self.g_centroids = self.g_repertoire.centroids[gameplay_mask][:, :2]
        self.g_genotypes = tree_util.tree_map(lambda x:x[gameplay_mask], self.g_repertoire.genotypes)

        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma
        self.logger = logger

        self.batch_size = batch_size
        self.scoring_fn = None     # scoring fn is set by setter method

    def set_scoring_fn(self, scoring_fn:HierarchyScore, *scoring_fn_args, **scoring_fn_kwargs) -> None:

        self.scoring_fn = scoring_fn(self, *scoring_fn_args, **scoring_fn_kwargs)
        if self.logger:
            self.logger.info('Set scoring fn!')
        
    @partial(jax.jit, static_argnames=("self",))
    def generate_genotypes(self, RNGKey: Array) -> Tuple[Array, Array]:

        key, subkey = jrand.split(RNGKey)
        structure_descriptors = jrand.uniform(subkey, (self.batch_size, 6)) 

        key, subkey = jrand.split(key)
        gameplay_descriptors = jrand.uniform(subkey, (self.batch_size, 2))

        genotype = structure_descriptors, gameplay_descriptors

        return genotype, key
    
    @partial(jax.jit, static_argnames=("self",))
    def score_genotypes(self, genotypes: Array, RNGKey: Array) -> Tuple[float | Array | None]:
        
        """
        Score genotype 

        """

        assert self.scoring_fn is not None, 'scoring fn not set!'

        key, subkey = jrand.split(RNGKey)
        keys = jrand.split(subkey, self.batch_size)
        fitness, bds = jax.vmap(self.scoring_fn)(genotypes, keys)

        return fitness, bds, None, key
    
    @partial(jax.jit, static_argnames=("self",))
    def variation_fn(self, x1, x2, RNGKey) -> Tuple[Array, Array]:
        
        """
        Isoline variation
        
        """
        
        y, key = isoline_variation(x1, x2, RNGKey, self.iso_sigma, self.line_sigma, 0, 1)

        return y, key
    
    @partial(jax.jit, static_argnames=("self",))
    def mutation_fn(self, x1, RNGKey) -> Tuple[Array, Array]:

        """
        Randomly increment of the BDs in one of the genotypes
        
        """
        return super().variation_fn(x1, RNGKey)
    
    @partial(jax.jit, static_argnames=("self",))
    def express(self, x) -> Array:

        """
        Genotype to phenotype mapping fn

        """

        structure_bds, gameplay_bds = x

        # gets nearest cell indicies 
        structure_cell_idxs = get_cells_indices(structure_bds, self.s_centroids)
        gameplay_cell_idxs = get_cells_indices(gameplay_bds, self.g_centroids)

        def retrieve_genotype(cell_idx, genotypes):
            return tree_util.tree_map(lambda x: jnp.squeeze(x[cell_idx], axis= 0), genotypes)
        
        structure_genotypes = jax.vmap(retrieve_genotype, in_axes=(0, None))(structure_cell_idxs, self.s_genotypes)
        gameplay_genotypes = jax.vmap(retrieve_genotype, in_axes=(0, None))(gameplay_cell_idxs, self.g_genotypes)

        structure_batch, start_end_batch = self.s_G.express_no_start_end(structure_genotypes)    
        level_batch = jax.vmap(self.g_G.express, in_axes=0)(gameplay_genotypes, structure_batch) * structure_batch
        level_batch = self.s_G.apply_start_end(start_end_batch, level_batch)
         
        return level_batch


@jax.jit
def get_cells_indices(
    batch_of_descriptors: jnp.ndarray, centroids: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns the array of cells indices for a batch of descriptors
    given the centroids of the repertoire.

    Args:
        batch_of_descriptors: a batch of descriptors
            of shape (batch_size, num_descriptors)
        centroids: centroids array of shape (num_centroids, num_descriptors)

    Returns:
        the indices of the centroids corresponding to each vector of descriptors
            in the batch with shape (batch_size,)
    """

    def _get_cells_indices(
        descriptors: jnp.ndarray, centroids: jnp.ndarray
    ) -> jnp.ndarray:
        """Set_of_descriptors of shape (1, num_descriptors)
        centroids of shape (num_centroids, num_descriptors)
        """
        return jnp.argmin(
            jnp.sum(jnp.square(jnp.subtract(descriptors, centroids)), axis=-1)
        )

    func = jax.vmap(lambda x: _get_cells_indices(x, centroids))
    return func(batch_of_descriptors)











        

    


        
    
    



    