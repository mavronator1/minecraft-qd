import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
import yaml
import jax.random as jrand
import logging
import time 
import numpy as np
import random
import os

from typing import Tuple, Callable, List
from qdax.utils.plotting import plot_2d_map_elites_repertoire, plot_multidimensional_map_elites_grid
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
# from evocraft_tools.block_utils import ClientHandler
from jax.flatten_util import ravel_pytree
from jax import Array
from common.genotype import Genotype

from gameplay.genotype import GameplayGenotype
from structure.genotype import StructureGenotype
from hierarchy.genotype import MiddleLayer
from flat.genotype import FlatGenotype

from jax.tree_util import tree_structure

from gameplay.scoring_fns import DistributionScore, GameplayScore, StructureDistributionScore, NewDistributionScore
from structure.scoring_fns import CVScore, Custom
from hierarchy.scoring_fns import BasicScore, GameTypeScore, PathFindScore, MaximizePath
from flat.scoring_fns import CombinedScore


GENOTYPES_DICT = {
     'structure': StructureGenotype,
     'gameplay': GameplayGenotype,
     'hierarchy': MiddleLayer,
     'flat': FlatGenotype
}

SCORING_FN_DICT  = {
    'structure': {
        'cv': CVScore,
        'custom': Custom,
        },

    'gameplay': {
        'sds': StructureDistributionScore,
        'ds': DistributionScore,
        'nds': NewDistributionScore
        },

    'hierarchy': {
        'basic': BasicScore,
        'gametype': GameTypeScore,
        'path': PathFindScore,
        'max_path': MaximizePath
    },
    'flat': {
        'max_path': MaximizePath,
        'combined': CombinedScore
    }
}


def load_genotype(task_name:str, **kwargs) -> Genotype:

    G = GENOTYPES_DICT.get(task_name.lower(), None)

    if G is None:
         raise NotImplementedError("Task not found!")
    return G(**kwargs)


def get_scoring_fn(task_name:str, scoring_fn_name:str) -> GameplayScore:


    scoring_fn = SCORING_FN_DICT.get(task_name.lower(), None).get(scoring_fn_name.lower(), None)
    if scoring_fn is None:
        raise NotImplementedError("Scoring fn not found!")
    
    return scoring_fn
    

def get_emitter_fns(G:Genotype) -> Tuple[Callable, Callable]:
    
    """
    Get valid mutation and emitter fns from Genotype
    
    """
    key = jrand.PRNGKey(0)

    dummy_x1, key = G.generate_genotypes(key)
    dummy_x2, key = G.generate_genotypes(key)

    try: 
         G.variation_fn(dummy_x1, dummy_x2, key)
    except NotImplementedError:
         variation_fn = None
    else:
         variation_fn = G.variation_fn
        
    try: 
          G.mutation_fn(dummy_x1, key)
    except NotImplementedError:
         mutation_fn = None
    else:
         mutation_fn = G.mutation_fn

    return mutation_fn, variation_fn


def load(repertoire_path:str, task_config_path:str) -> Tuple[MapElitesRepertoire, Genotype]:
    
    """
    Loads a saved repertoire, returning the repertoire + the Genotype class for that repertoire
    
    """

    with open(task_config_path) as f:
        config = yaml.safe_load(f)

    G = load_genotype(config['name'], **config['genotype_setup'], batch_size=1)

    key = jrand.PRNGKey(0)
    dummy_genotype, _ = G.generate_genotypes(key)

    _, reconstruction_fn = ravel_pytree(dummy_genotype)
    repertoire = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=os.path.join(repertoire_path))

    return repertoire, G


def random_structure_sample(repertoire:MapElitesRepertoire, G:StructureGenotype, n:int, RNGKey:Array) -> dict:

    key, subkey = jrand.split(RNGKey)
    idxs = random_sample(repertoire, n, subkey)
    structures = []

    for idx in idxs:
        genotype = jax.tree_util.tree_map(lambda x: x[idx], repertoire.genotypes)
        structure = G.express(genotype)[0, :, :]
        idx_dict = {
            'structure': structure,
            'BDs': repertoire.descriptors[idx],
            'Fitness': repertoire.fitnesses[idx],
            'idx': idx
        }
        structures.append(idx_dict)

    return structures

def random_gameplay_sample(repertoire:MapElitesRepertoire, G:GameplayGenotype, n:int, RNGKey:Array, structures:list[Array]=None):

    key, subkey = jrand.split(RNGKey)
    idxs = random_sample(repertoire, n, subkey)
    gameplay_networks = []

    for idx in idxs:
        genotype = jax.tree_util.tree_map(lambda x: x[idx], repertoire.genotypes)

        idx_dict = {
            'network': lambda x: G.network.apply(genotype, x),
            'expressed': [],
            'BDs': repertoire.descriptors[idx],
            'Fitness': repertoire.fitnesses[idx],
            'idx': idx
        }
        
        if structures is not None:
            for i, structure in enumerate(structures):
                # idx_dict['expressed'].append(G.express(genotype, structure)[0, :, :]) # when G.express was batched...
                idx_dict['expressed'].append(G.express(genotype, structure))

        gameplay_networks.append(idx_dict)

    return gameplay_networks
        
def random_sample(repertoire:MapElitesRepertoire, n:int, RNGKey:Array) -> List:

    """
    Returns n random individuals from the repertoire by their idx 
    
    """

    # get the indexes of all filled cells in the repertoire 
    idxs = jnp.array([idx for fitness, idx in sorted(zip(repertoire.fitnesses, range(len(repertoire.fitnesses))), 
                                        key=lambda x: x[0], reverse=True) 
                                        if fitness != -jnp.inf])
    
    assert n <= len(idxs), "n must be < individuals in the repertoire" 

    # randomly select some values
    if n < 1:
        n = len(idxs) 
    
    key, subkey = jrand.split(RNGKey)
    samples = np.array(jrand.choice(subkey, idxs, (n, ), replace=False))
    return samples    


    

        
    
        

        
        
    
        

