"""
Fitness functions for NEAT methods for evolving gameplay operators

"""

import jax
import jax.numpy as jnp

from logging import Logger
from abc import ABC, abstractmethod
from jax import Array, vmap, lax
from typing import List
from functools import partial


class GameplayFitnessFunction:

    """
    Base class for gameplay fitness functions

    """

    n_placeable_features = None # number of placeable ( > 2) features

    def __init__(self):
        pass

    @abstractmethod
    def score(self, structure:Array, prediction:Array, logger:Logger=None) -> float:
        """
        Calculates and returns fitness
        """
        pass

    def __call__(self, structure:Array, prediction:Array, logger:Logger=None):
        return self.score(structure, prediction, logger)


class CombinedFitness(GameplayFitnessFunction):

    """
    Combine multiple fitness functions together 
    
    """

    def __init__(self, fitness_fns:List[GameplayFitnessFunction], weights:List[float]):
        
        super().__init__()
        self.fitness_fns = fitness_fns
        self.weights = weights

    def score(self, structure:Array, prediction:Array, logger:Logger=None):
        score = 0
        for fn, w in zip(self.fitness_fns, self.weights):
            score += w*fn(structure, prediction,logger)
        return score




class FrequencyFitness(GameplayFitnessFunction):

    """
    Scores how close gameplay features are to a target amount

    """

    def __init__(self, targets:List):

        """
        targets: expected number of gameplay elements, idx 0 = element 2, idx = 1 = element 3 etc.. 
        
        """

        super().__init__()
        assert len(targets) == self.__class__.n_placeable_features, 'targets must equal number of positive gameplay features!'

        self.targets = jnp.array(targets)

    def score(self, structure:Array, prediction:Array, logger:Logger=None):
        
        def scan_f(carry, x):

            freq = jnp.sum(carry==x)
            return carry, freq
        
        target_vals = jnp.arange(2, self.targets.shape[0]+2)
        carry, freqs = lax.scan(scan_f, prediction, target_vals)

        sq_diff = jnp.square(freqs - self.targets)
        score = 1 - (jnp.sum(sq_diff) / self.targets.shape[0])

        if logger:
            logger.info(f"sq_diff: {sq_diff}")
            logger.info(f"freq score: {score}")

        return score

        
class AccuarcyFitness(GameplayFitnessFunction):

    """

    Scores how much of the gameplay elements are in the playable area of the structure. 
    Encourages the number of gameplay elements by having number of elements close to
    density. 

    """

    def __init__(self, accuracy_w:float, inaccuracy_w:float, density_target:float, density_w:float):

        self.accuracy_w = accuracy_w 
        self.inaccuracy_w = inaccuracy_w
        self.density_target = density_target # overall gameplay feature density
        self.density_w = density_w


    def score(self, structure:Array, prediction:Array, logger:Logger=None):

        canvas_size = structure.shape[0] ** 2
        structure_area = jnp.sum(structure)
        empty_area = jnp.sum(structure==0)

        # network correctly places things

        valid_placements = (jnp.sum(jnp.logical_and(structure==1, prediction>=1)) + 
                            jnp.sum(jnp.logical_and(structure==0, prediction==0)))
        
        placement_accuracy = valid_placements/canvas_size

        # network incorrectly places things

        invalid_placements = jnp.sum(jnp.logical_and(structure==0, prediction>=1))
        placement_inaccuracy = invalid_placements/empty_area

        # density target 

        placement_density = jnp.sum(jnp.logical_and(structure==1, prediction>=1)) / structure_area
        density  = (self.density_target - placement_density)

        if logger:
            logger.info(f'placement_accuracy: {placement_accuracy}\n')
            logger.info(f'placement_inaccuracy: {placement_inaccuracy}\n')
            logger.info(f'density: {density}\n')

        score = (self.accuracy_w*placement_accuracy - 
                 self.inaccuracy_w*placement_inaccuracy  -
                 self.density_w*abs(density))

        return score
    

FNS_DICT = {
    'accuracy': AccuarcyFitness,
    'frequency': FrequencyFitness
}

def load_fitness_fn(config):
    
    # to do: compose multiple fns together

    fns = []
    weights = []

    for entry in config['functions']:
        fn_name = entry['name']
        weight = entry['w']
        params = entry.get('parameters', {})
        if fn_name in FNS_DICT:
            f = FNS_DICT[fn_name](**params)
            fns.append(f)
            weights.append(weight)
        else:
            raise ValueError(f"Fitness function '{fn_name}' not found.")
    
    if len(fns) > 1:
        return CombinedFitness(fns, weights)
    else:
        fns[0]

    
