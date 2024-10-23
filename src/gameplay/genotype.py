"""
Gameplay genotype for MAP-Elites 

"""

import jax 
import jax.numpy as jnp
import jax.random as jrand 
import jax.nn as jnn
import scipy.stats as stats


from jax import vmap, lax, Array
from functools import partial
from typing import Tuple, List
from itertools import product
from common.genotype import Genotype
from functools import partial
from logging import Logger
from gameplay.scoring_fns import GameplayScore

from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation


class GameplayGenotype(Genotype):

    """
    
    
    """

    def __init__(self,
                 
                 window_size:int,
                 num_features:int,
                 layer_sizes:Tuple[int],
                 batch_size: int,

                 iso_sigma: float,
                 line_sigma: float,

                 input_middle = False, 
                 logger = None
                ):
        
        """
        window_size {int}: window_size x window_size mask that 
        contains gameplay features 
        num_features {int}: number of gameplay features 
        layer_sizes {Tuple[int]}: hidden layer sizes
        target_p {float}: target p of placing a gameplay element
        input_middle {bool}: if to input the middle value of the array

        """

        self.window_size = window_size 
        self.input_size = (window_size ** 2) 
        self.input_middle = input_middle

        if not input_middle:
            self.input_size -= 1
    
        self.num_features  = num_features
        self.layer_sizes = layer_sizes

        self.batch_size = batch_size

        self.logger = logger

        if self.logger:
            self.logger.info("Gameplay Genotype")

        self.network = MLP(
            layer_sizes= self.layer_sizes + [self.num_features+1],
            final_activation=jnn.softmax,
            kernel_init=jnn.initializers.xavier_normal()
        )   

        self.scoring_fn = None

        self.variation_fn = partial(isoline_variation, 
                                    iso_sigma = iso_sigma, 
                                    line_sigma = line_sigma)  
        
    def set_scoring_fn(self, scoring_fn:GameplayScore, *scoring_fn_args, **scoring_fn_kwargs) -> None:
        self.scoring_fn = scoring_fn(self, *scoring_fn_args, **scoring_fn_kwargs)
        if self.logger:
            self.logger.info('Set scoring fn!')

    @partial(jax.jit, static_argnames=("self",))
    def generate_genotypes(self, RNGKey) -> Tuple[dict, Array]:

        """
        Init and return variables for MLP genotype
        
        """
        key, subkey = jrand.split(RNGKey)
        keys = jrand.split(subkey, self.batch_size)

        empty_batch = jnp.empty((self.batch_size, self.input_size))
        init_variables = jax.vmap(self.network.init)(keys, empty_batch)

        return init_variables, key
    
    @partial(jax.jit, static_argnames=("self",))
    def score_genotypes(self, genotypes: Array, RNGKey: Array) -> Tuple[float, Array, None, Array]:
        
        """
        Score a gameplay network
        
        """

        assert self.scoring_fn is not None, 'gameplay_score not set!'

        key, subkey = jrand.split(RNGKey)
        keys = jrand.split(subkey, self.batch_size)

        fitnesses, bds = jax.vmap(self.scoring_fn)(genotypes, keys)
        
        return fitnesses, bds, None, key
        
    @partial(jax.jit, static_argnames=("self",))
    def variation_fn(self, x1, x2, RNGKey):
        return self.variation_fn(x1, x2, RNGKey)
    
    @partial(jax.jit, static_argnames=("self",))
    def mutation_fn(self, x1, RNGKey):
        raise NotImplementedError

    @partial(jax.jit, static_argnames=("self"))
    def express(self, x:dict, structure:Array):

        """
        Applies a network over a structure
        
        """ 
        padded_level, padding = self.pad_level(structure)

        render_fn = partial(self.scan_render_gameplay, 
                            structure=padded_level, 
                            start_idx=padding,
                            end_idx=structure.shape[0],
                            unpadded_size=structure.shape[0])
                
        network_output = render_fn(variables=x)  

        level = jnp.where(jnp.greater(network_output, structure), 
                          network_output,
                          structure)
        
        return level

    @partial(jax.jit, static_argnames=("self", "unpadded_size"))
    def scan_render_gameplay(self, structure, variables, start_idx, end_idx, unpadded_size):
    
        """

        Uses scan to apply network to level to add gameplay elements

        """

        carry = structure, variables, (start_idx, start_idx), start_idx, end_idx

        # NOW ONLY WORKING FOR INPUT_MIDDLE = TRUE

        def scan_f(carry, x):

            # unpack
            level, variables, p, start_idx, end_idx = carry

            new_line = lambda p: (p[0]+1, start_idx)
            reset_cond = p[1] == end_idx
            p = lax.cond(reset_cond, new_line, lambda x: x, p) # reset line if needed
            x = lax.dynamic_slice(level, (p[0]-1, p[1]-1), (self.window_size, self.window_size)) # take slice

            x = jnp.ravel(x) # flatten   
            # x = jnp.delete(x, (self.window_size**2) // 2) # remove middle value
          
            y = jnp.argmax(self.network.apply(variables, x), axis=0) # fwd pass
            y += y != 0 # make sure y is never 1 (structure value)

            level = lax.cond(y != 0, 
                             lambda y: level.at[p].set(y), 
                             lambda y: level, 
                             y)  

            p = (p[0], p[1]+1) # increment cols

            carry = level, variables, p, start_idx, end_idx

            return carry, y
        
        (level, _, _, _, _), y = lax.scan(scan_f, carry, None, unpadded_size**2)
        y = jnp.reshape(y, (unpadded_size, unpadded_size))
        return y

    @partial(jax.jit, static_argnames=("self",))
    def pad_level(self, levels:Array) -> Tuple[Array, int]:

        """
        Applies required padding for levels given the window_size
        
        Args:
            levels: m x m level
        """
 
        padding =  (self.window_size - 1) // 2
        padded_levels = jnp.pad(levels, 
                                ((padding, padding), (padding, padding)))
        
        return padded_levels, padding



















