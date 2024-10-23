from abc import ABC, abstractmethod
from jax import Array
from typing import Tuple, Any


class Genotype(ABC):
    
    """
    Base class for genotypes in MAP-Elites 
    
    """

    @abstractmethod
    def generate_genotypes(self, RNGKey: Array) -> Tuple[Any, Array]:
        pass

    @abstractmethod
    def set_scoring_fn(self, fn, *args) -> Any:
        pass

    @abstractmethod
    def score_genotypes(self, genotype:Array, RNGKey: Array) -> Tuple[float, Array, None, Array]:
        pass

    @abstractmethod
    def variation_fn(self, x1, x2, RNGKey) -> Tuple[Any, Array]:
        pass

    @abstractmethod
    def mutation_fn(self, x1, RNGKey) -> Tuple[Any, Array]:
        pass

    @abstractmethod
    def express(self, x, *args) -> Any:
        pass







