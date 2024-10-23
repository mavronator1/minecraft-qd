import jax
import jax.numpy as jnp


from typing import Tuple
from jax import Array, lax 
from functools import partial


class JaxQueue:

    def __init__(self, max_size, num_cols, dtype=jnp.int16, batch_size=4, stack=False) -> Array:

        self.max_size = max_size
        self.num_cols = num_cols
        self.Q_dtype = dtype
        self.batch_size = batch_size 

        if stack:
            self.order_inc = -1
        else:
            self.order_inc = 1
        
        print(self.order_inc)

        
    @partial(jax.jit, static_argnames=("self",))
    def new_queue(self) -> Tuple[Array, Array]:
        """
        Returns empty queue and order
        """
        # +1 cols for the queue position
        return jnp.full((self.max_size, self.num_cols+1), jnp.inf), 0

    @partial(jax.jit, static_argnames=("self",))
    def insert(self, Q:Array, Q_order:int, val:Array) -> None: 

        assert val.shape[0] == self.num_cols
        
        new_row = jnp.hstack((val, Q_order))

        Q_order += self.order_inc
        Q = Q.at[-1].set(new_row)
        Q = self.sort_Q(Q)

        return Q, Q_order
    
    @partial(jax.jit, static_argnames=("self",))
    def batch_insert(self, Q:Array, Q_order:int, vals:Array) -> None:

        assert vals.shape[1] == self.num_cols

        vals_mask = jnp.any(~jnp.isinf(vals), axis=1) 
        order_batch = jnp.where(vals_mask, jnp.array([Q_order]), jnp.array([jnp.inf])) 
        vals = jnp.hstack([vals, order_batch[:, jnp.newaxis]])

        Q = Q.at[(-1-self.batch_size):-1].set(vals)
        Q = self.sort_Q(Q)
        Q_order += self.order_inc

        return Q, Q_order

    @partial(jax.jit, static_argnames=("self",))
    def pop(self, Q:Array) -> Tuple[Array, Array]:

        pop_value = Q[0]
        Q = Q.at[0].set(jnp.full((self.num_cols+1), jnp.inf))
        Q = self.sort_Q(Q)

        return Q, pop_value
    
    @partial(jax.jit, static_argnames=("self",))
    def sort_Q(self, Q:Array):
        col_sort = jnp.argsort(Q[:, -1])
        return Q[col_sort]
    

    @partial(jax.jit, static_argnames=("self",))
    def len(self, Q:Array):
        return jnp.sum(Q != jnp.inf)

    
class JaxList:

    def __init__(self, max_size, num_cols, batch_size=4):
        
        self.max_size = max_size
        self.num_cols = num_cols
        self.batch_size = batch_size

    @partial(jax.jit, static_argnames=("self",))
    def new_list(self):
        """
        Returns new list and its size
        """
        return jnp.full((self.max_size, self.num_cols), jnp.inf), 0

    @partial(jax.jit, static_argnames=("self",))
    def append(self, list:Array, list_size:int, val:Array) -> Tuple[Array, Array]:

        assert val.shape[0] == self.num_cols

        list = list.at[list_size].set(val)
        list_size += 1

        return list, list_size
    
    @partial(jax.jit, static_argnames=("self",))
    def batch_append(self, list:Array, list_size:int, vals:Array) -> Tuple[Array, Array]:

        """
        Num directions is batch size 
        
        """

        assert vals.shape[1] == self.num_cols, f"{vals.shape[1]} is not valid!"

        mask = jnp.isinf(vals).all(axis=1)
        sort_mask = jnp.lexsort((vals[:, 1], vals[:, 0], mask))
        vals = vals[sort_mask]

        num_vals = jnp.sum(jnp.all(vals != jnp.inf, axis=1))

        list = lax.dynamic_update_slice(list, vals, (list_size, self.batch_size))

        list_size += num_vals

        return list, list_size
        

    @partial(jax.jit, static_argnames=("self",))
    def is_in(self, list:Array, val:Array) -> bool:

        assert val.shape[0] == self.num_cols, f"{val.shape[0]} is not valid!"

        return jnp.any(jnp.all(val == list, axis=1))
