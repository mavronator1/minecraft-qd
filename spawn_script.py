"""
Script for spawning in Minecraft 

"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import jax.numpy as jnp

from evocraft_tools.spawn_fns import hbr_spawn_random, clear_area, explore_hbr, explore_structure, explore_hbrs, explore_bd, spawn_from_array
from evocraft_tools.block_utils import ClientHandler
from evocraft_tools.minecraft_pb2 import *

if __name__ == "__main__":

    jnp.set_printoptions(2)

    # explore_structure(
    #     "results/structure/5BD/",
    #     "results/structure/5BD/structure.yaml"
    #  )

    hbr_spawn_random("results/hierarchy/final/", "results/hierarchy/final/hierarchy.yaml", 1.0, 2000, start_x=0)

    # explore_hbrs(
    #     "results/hierarchy/final/", "results/hierarchy/final/hierarchy.yaml",
    #     "results/flat/with-structure/", "configs/tasks/flat.yaml",
    #     spawn_point = (-800, 0)
    # )

    # hbr_spawn_random("output/2024-08-28/11-13-50/flat/repertoire/","configs/tasks/flat.yaml", 1.0, 12, 0, 4)
    # hbr_spawn_random("output/2024-08-28/09-48-36/hierarchy/repertoire/", "configs/tasks/hierarchy.yaml", 1.0, 12, 130, 4)

    # explore_bd(
    #     "src/structure/v3/",
    #     "src/structure/v3/structure.yaml",
    #     4,
    #     jnp.array([10 / 20, 1 / 20, 1 / 20, 12 / 20]),
    # )
 

    # s = ClientHandler()
    # s.add_block((0, 4, 0), NORTH, FLOWING_LAVA)
    # s.send_to_server()


   

    

# 57070
