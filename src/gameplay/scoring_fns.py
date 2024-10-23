"""
Scoring fns for the Gameplay Genotype

"""
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

from typing import Tuple, TYPE_CHECKING
from logging import Logger
from jax.tree_util import tree_map
from collections import defaultdict


if TYPE_CHECKING:
    from gameplay.genotype import GameplayGenotype
    from common.utils import load

from jax import lax, Array
from itertools import product
from functools import partial
from common.scoring_fn import ScoringFn

class GameplayScore(ScoringFn):

    def __init__(self, G:'type[GameplayGenotype]'):

        self.window_size = G.window_size
        self.input_size = G.input_size
        self.num_features  = G.num_features
        self.network = G.network

        self.logger = G.logger


class NewDistributionScore(GameplayScore):

    """
    Uses all possible valid combinations (num_features ** num_inputs) as training data. 
    Also assuems genotype takes 9 inputs, not 8! 
    
    BD: distribution of every gameplay element across entire training set 
    Fitness: error of target number of 0s to number of 0s generated

    """

    def __init__(self, G:'type[GameplayGenotype]', target_zero_rate:float, target_zero_tolerance:float, priority_weight:float,
                 structure_repertoire:str=None, structure_config:str=None):

        super().__init__(G)

        self.target_zero_rate = target_zero_rate
        self.target_zero_tolerance = target_zero_tolerance
        self.priortiy_weight = priority_weight

        train_x = self.generate_training_data()

        # higher weight at i means a zero at train_x[i] is more important
        # e.g. zero_weights[i] = 4 means it being a zero is worth 4x more in zero rate calculation
        self.zero_weights = self.generate_fitness_weights(train_x, structure_repertoire, structure_config)

        self.train_x = jnp.reshape(train_x, (train_x.shape[0], self.input_size))

    @partial(jax.jit, static_argnames=("self",))
    def score(self, genotype: Array, RNGKey: Array) -> Tuple[float, Array]:

        y = jax.vmap(self.network.apply, in_axes=(None, 0))(genotype, self.train_x)

        y = jnp.argmax(y, axis=1)
        y += y != 0 # prevents 1 being output 
        
        accuracy_score = 1 - jnp.sum(jnp.logical_and(self.train_x[:, 4] == 0, y > 1)) / jnp.sum(self.train_x[:, 4]==0)
        
        zero_rate = jnp.minimum(jnp.sum(jnp.where(y==0, self.zero_weights, 0)) / len(self.train_x), 1)
        deviation = jnp.abs(zero_rate - self.target_zero_rate)
        zero_rate_score = jnp.where(jnp.less_equal(deviation, self.target_zero_tolerance),
                                    1,
                                    1 - (deviation / self.target_zero_rate))

        fitness = (accuracy_score + zero_rate_score) / 2
        
        _, counts =  jnp.unique(y, return_counts=True, size=self.num_features+1)
        bds = counts[1:] / jnp.sum(counts[1:])
        bds = bds[:2]

        return fitness, bds

    def generate_fitness_weights(self, train_x, structure_repertoire, structure_config):

        """
        Generates certain weights to increase importance on data more likely to appear

        """

        fitness_weights = jnp.full(len(train_x), 1)

        if structure_repertoire is None:

            m1 = jnp.all(train_x == 0, axis=(1, 2)) 
            m2 = jnp.all(train_x == 1, axis=(1, 2))
            m3 = jnp.sum(train_x == 1, axis=(1, 2)) == 1
            m4 = jnp.sum(train_x == 1, axis=(1, 2)) == 2

            mask = m1 | m2 | m3 | m4
        
        else:

            self.logger.info("Generating patch distribution")
            patch_distribution = self.generate_patch_distribution(structure_repertoire, structure_config)
            

        fitness_weights = jnp.where(mask, self.priortiy_weight, fitness_weights)

        return fitness_weights
    
    def generate_training_data(self):

        """
        Generates the training data
        """

        if self.logger:
            self.logger.info("Generating training data")

        train_x = jnp.array(list(product(range(self.num_features+2), 
                                          repeat=self.input_size))) # expected 1953125 x 9

        train_x = jnp.reshape(train_x, (train_x.shape[0], int(jnp.sqrt(self.input_size)), int(jnp.sqrt(self.input_size))))                                  
        
        mask = jnp.array(
            (train_x[:, 0, 2] == 2) | (train_x[:, 0, 2] == 3) | (train_x[:, 0, 2] == 4) |
            (train_x[:, 1, 2] == 2) | (train_x[:, 1, 2] == 3) | (train_x[:, 1, 2] == 4) |
            (train_x[:, 2, 0] == 2) | (train_x[:, 2, 0] == 3) | (train_x[:, 2, 0] == 4) |
            (train_x[:, 2, 1] == 2) | (train_x[:, 2, 1] == 3) | (train_x[:, 2, 1] == 4) |
            (train_x[:, 2, 2] == 2) | (train_x[:, 2, 2] == 3) | (train_x[:, 2, 2] == 4) |
            (train_x[:, 1, 1] == 2) | (train_x[:, 1, 1] == 3) | (train_x[:, 1, 1] == 4)
        )

        train_x = train_x[~mask]

        if self.logger:
            self.logger.info(f"{train_x.shape[0]} samples generated!")

        return train_x

    def generate_patch_distribution(self, structure_repertoire, structure_config):

        sr, sg = load(structure_repertoire, structure_config)

        valid_idx = jnp.argwhere(sr.fitnesses != -jnp.inf)
        idx_sample = jrand.choice(jrand.PRNGKey(0), valid_idx, (1000,))
        patch_distribution = defaultdict(float)

        for i, idx in enumerate(idx_sample):
            genotype = tree_map(lambda x: lax.collapse(x[idx], 0, 2), sr.genotypes)
            structure, _ = sg.express_no_start_end(genotype)
            structure = structure[0]
            padded_structure = jnp.pad(structure, ((1, 1), (1, 1)))
            patches = self.G.scan_render_gameplay(padded_structure, 1, 20, 20)
            structure_distribution = defaultdict(int)    
            
            for patch in patches:

                if patch[4] == 0:
                    continue

                np_patch = np.array(patch)
                tuple_patch = tuple(np_patch)
                structure_distribution[tuple_patch] += 1
                
            for patch, freq in structure_distribution.items():
                patch_distribution[patch] += (freq / 400)

        for patch in patch_distribution:
            patch_distribution[patch] /= 1000
        
        return patch_distribution


class StructureDistributionScore(GameplayScore):
    
    """

    Uses only possible structure data (1s and 0s so 2 ** num_inputs) as training data.

    BD: distribution of every gameplay element across entire training set 
    Fitness: error of target number of 0s to number of 0s generated

    """

    def __init__(self, G:'type[GameplayGenotype]', target_0:float):
        super().__init__(G)

        self.target_0 = target_0

        if self.logger:
            self.logger.info("Generating training data")

        self.train_x =  jnp.array(list(product(range(2), repeat=self.input_size)))

        if self.logger:
            self.logger.info(f"{self.train_x.shape[0]} samples generated!")

    
    @partial(jax.jit, static_argnames=("self",))
    def score(self, genotype: Array, RNGKey: Array) -> Tuple[float, Array]:

        y = jax.vmap(self.network.apply, in_axes=(None, 0))(genotype, self.train_x)

        y = jnp.argmax(y, axis=1)
        y += y != 0 # prevents 1 being output 

        _, counts =  jnp.unique(y, return_counts=True, size=self.num_features+1)
        
        bds = counts[1:] / sum(counts[1:]) # bds is distribution of features

        fitness = 1 - jnp.abs(self.target_0 - (counts[0]/jnp.sum(counts)))

        return fitness, bds


class DistributionScore(GameplayScore):

    """
    Uses all possible combinations (num_features ** num_inputs) as training data
    
    BD: distribution of every gameplay element across entire training set 
    Fitness: error of target number of 0s to number of 0s generated

    """

    def __init__(self, G:'type[GameplayGenotype]', target_0:float):
        super().__init__(G)

        self.target_0 = target_0

        self.train_x = self.generate_training_data()

    
    @partial(jax.jit, static_argnames=("self",))
    def score(self, genotype: Array, RNGKey: Array) -> Tuple[float, Array]:

        y = jax.vmap(self.network.apply, in_axes=(None, 0))(genotype, self.train_x)

        y = jnp.argmax(y, axis=1)
        y += y != 0 # prevents 1 being output 

        

        values, counts =  jnp.unique(y, return_counts=True, size=self.num_features+1)
        bds = counts[1:] / jnp.sum(counts[1:])

        fitness = 1 - jnp.abs(self.target_0 - (counts[0]/jnp.sum(counts)))

        return fitness, bds


    def generate_training_data(self):

        """
        Generates the training data
        """

        if self.logger:
            self.logger.info("Generating training data")

        train_x = jnp.array(list(product(range(self.num_features+2), 
                                          repeat=self.input_size)))
    
        if self.logger:
            self.logger.info(f"{train_x.shape[0]} samples generated!")

        return train_x





