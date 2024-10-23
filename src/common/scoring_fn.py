import jax
import jax.numpy as jnp
import jax.random as jrand

from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

from jax import lax, Array
from itertools import product
from functools import partial
from common.genotype import Genotype


class ScoringFn(ABC):

    @abstractmethod
    def score(self, genotype:Array, RNGKey: Array) -> Tuple[float, Array]:
        """
        Scores a single genotype, returing the fitness and BD

        """
        raise NotImplementedError

    def __call__(self, genotype:Array, RNGKey: Array) -> Tuple[float, Array]:
        return self.score(genotype, RNGKey)

      

    
