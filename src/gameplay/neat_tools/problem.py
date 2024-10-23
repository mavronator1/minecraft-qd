"""
Gameplay Problem for NEAT genotype

"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import os

from jax.nn import one_hot
from jax import vmap, lax, Array
from typing import Callable, Tuple
from tensorneat.common import State, StatefulBaseClass
from tensorneat.problem import BaseProblem
from tensorneat.genome import DefaultGenome
from functools import partial
from common.utils import load_repertoire, random_sample
from gameplay.fitnesses import GameplayFitnessFunction

class GameplayProblem(BaseProblem):

    """
    Evaluates NNs for the gameplay problem task  
    
    """
    
    jitable = True

    def __init__(self, 
                 
                 fitness_fn:GameplayFitnessFunction, # fitness function to use
                 
                 n_tiles:int, # number of tiles that can be generated 
                 window_size:int, # context window applied around each point (n in n x n)
                 n_levels:int, # number of levels to sample 

                 RNGKey:Array,
                 repertoire_path:str, # path of structure repertoire to sample levels from
                 structure_config:str='configs/tasks/structure.yaml', # path of config for structure genotype
                 logger = None
                 ):
        
        assert window_size % 2 == 1, "window_size must be odd"

        self.fitness_fn = fitness_fn

        self.n_tiles = n_tiles # num of gameplay tiles that can be generated 
        self.window_size = window_size # size of context window
        self.n_levels = n_levels # number of levels to sample

        if logger:
            logger.info("Loading repertoire...")

        repertoire, self.structure = load_repertoire(repertoire_path, structure_config)
        self.levels, _ = random_sample(repertoire, self.structure, RNGKey, n_levels)

        # possibly should process the data here if there is overfitting...

        self.train_x, self.padding = self.pad_level(self.levels)


    
    def evaluate(self, state: State, randkey, act_func:Callable, params):

        """
        Evaluates a single individual from the population

        """

        gen_fn = partial(self.scan_render_gameplay, 
                         start_idx=self.padding, 
                         end_idx=self.structure.size,
                         state=state,
                         act_func=act_func,
                         params=params) 

        predictions, _ = vmap(gen_fn, in_axes=0)(level=self.train_x) # n x 20 x 20

        fitnesses = vmap(self.fitness_fn, in_axes=(0, 0))(self.levels, predictions)

        return jnp.sum(fitnesses) / len(self.levels)


    def show(self, state:State, randkey, act_func:Callable, params, *args, **kwargs):
        
        """
        Show how a genome perform in this problem
        
        """

        logger = kwargs.get('logger', None)

        gen_fn = partial(self.scan_render_gameplay, 
                    start_idx=self.padding, 
                    end_idx=self.structure.size,
                    state=state,
                    act_func=act_func,
                    params=params) 

        predictions, _ = vmap(gen_fn, in_axes=0)(level=self.train_x) 

        if logger:
            for i in range(self.n_levels):

                logger.info(f'Structure mask: \n{self.levels[i]}')
                logger.info(f'Gameplay mask: \n{predictions[i]}')
                logger.info(f'Level: \n{self.apply_gameplay_mask(self.levels[i], predictions[i])}')
                logger.info(f'Fitness: \n{self.fitness_fn(self.levels[i], predictions[i], logger=logger)}')


    # @partial(jax.jit, static_argnames=("self",))
    def scan_render_gameplay(self, state, params, act_func, level, start_idx, end_idx):
        """

        Uses scan to apply network to level to add gameplay elements

        """

        carry = level, (start_idx, start_idx), start_idx, end_idx

        def scan_f(carry, x):

            # unpack
            level, p, start_idx, end_idx = carry

            new_line = lambda p: (p[0]+1, start_idx)
            reset_cond = p[1] > end_idx
            p = lax.cond(reset_cond, new_line, lambda x: x, p) # reset line if needed

            x = lax.dynamic_slice(level, (p[0]-1, p[1]-1), (self.window_size, self.window_size)) # take slice
            x = jnp.ravel(x) # flatten
            x = jnp.delete(x, (self.window_size**2) // 2) # remove middle value 
            x = lax.collapse(one_hot(x, self.n_tiles), 0)

            y = jnp.argmax(act_func(state, params, x), axis=0) # fwd pass
            y += y != 0 # make sure y is never 1 (structure value)

            level = lax.cond(y != 0, 
                             lambda y: level.at[p].set(y), 
                             lambda y: level, 
                             y)  

            p = (p[0], p[1]+1) # increment cols

            carry = level, p, start_idx, end_idx

            return carry, y
        
        (level, _, _, _), y = lax.scan(scan_f, carry, None, self.structure.size**2)
        y = jnp.reshape(y, (self.structure.size, self.structure.size))
        return y, level
    
    @partial(jax.jit, static_argnames=("self",))
    def pad_level(self, levels:Array) -> Tuple[Array, int]:

        """
        Applies required padding for levels given the window_size
        
        Args:
            levels: n x m x m levels 
        """
 
        padding =  (self.window_size - 1)//2
        padded_levels = jnp.pad(levels, 
                                ((0,0), (padding, padding), (padding, padding)))
        
        return padded_levels, padding


    def apply_gameplay_mask(self, level:Array, mask:Array) -> Array:
        """
        Applies the gameplay_mask to the level 
        
        """
        return jnp.where(mask != 0, mask, level)

    @partial(jax.jit, static_argnames=("self",))
    def level_to_slices(self, training_levels:Array, RNGKey:Array):

        """
        Transforms entire level into window size slices.

        """ 

        padded_levels, padding = self.pad_level(training_levels)
        
        patched_levels = jax.vmap(self.get_level_slices, in_axes=(0, None))(padded_levels, padding) 
        patched_levels = lax.collapse(patched_levels, 0, 2)

        one_hot_encoded = jnp.eye(self.n_tiles)[patched_levels]
        one_hot_encoded = lax.collapse(one_hot_encoded, 1, 3)

        return patched_levels
    
    @partial(jax.jit, static_argnames=("self",))
    def get_level_slices(self, level:Array, padding:int):
        """
        Extracts self.window_size x self.window_size patches from padded level,
        flattening and removing the centre value. 

        Args:
            level - n x n level with padding 
            padding - amount of padding added to edges 
        
        """

        start_idx = padding
        end_idx = level.shape[0] - start_idx - 1
        
        def scan_f(carry, x):

            c = carry
            new_line = lambda c: (c[0]+1, start_idx)
            reset_cond = c[1] > end_idx
            c = lax.cond(reset_cond, new_line, lambda c: c, c) # reset line if needed
            
            y = lax.dynamic_slice(level, (c[0]-1, c[1]-1), (self.window_size, self.window_size)) # take slice
            y = jnp.ravel(y) # flatten
            y = jnp.delete(y, (self.window_size**2) // 2) # remove middle value 

            c = (c[0], c[1]+1) # increment cols

            return c, y
        
        carry = (start_idx, start_idx)
        _, patches = lax.scan(scan_f, carry, None, length=self.structure.size**2)
        return patches
 
    @property 
    def input_shape(self):
        return self.n_levels, (self.window_size**2 - 1) * self.n_tiles
    
    @property
    def output_shape(self):
        return self.n_levels, 1


if __name__ == "__main__":
    print("hello!")