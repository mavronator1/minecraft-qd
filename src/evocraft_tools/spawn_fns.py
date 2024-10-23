import jax
import jax.random as jrand
import jax.numpy as jnp
import argparse
import logging 
import yaml
import time 
import random
import qdax.core.containers.mapelites_repertoire as me_repertoire

from evocraft_tools.block_utils import ClientHandler, AIR, COBBLESTONE, EMERALD_BLOCK, REDSTONE_BLOCK, CHEST, MOB_SPAWNER, TNT
from structure.genotype import StructureGenotype
from jax.flatten_util import ravel_pytree
from common.utils import load, random_sample, random_structure_sample, random_gameplay_sample
from gameplay.genotype import GameplayGenotype
from jax import lax, Array
from jax.tree_util import tree_map
from typing import Tuple

from evocraft_tools import minecraft_pb2, minecraft_pb2_grpc
# from evocraft_tools.minecraft_pb2 import _globals
import grpc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def spawn_from_array(array:Array):

    """
    Spawn from an array

    """

    server = ClientHandler() # minecraft server 
    clear_area(server, (0, 20), (4, 10), (0, 20))
    server.spawn_structure(array, 0, 4, 0)





def spawn_from_idx(repertoire_path, config_path):
    """
    Spawn by inputting idxs
    
    """

    logging.info('Loading Repertoire...')
    repertoire, G = load(repertoire_path, config_path)
    server = ClientHandler() # minecraft server 
    clear_area(server, (0, 500), (4, 10), (0, 50))

    x = 0
    x_delta = 30
    while True:
        idx = int(input("input idx to enter: "))

        genotype = tree_map(lambda x : x[idx], repertoire.genotypes)
        print(genotype[0].shape)
        phenotype = G.express(genotype)[0]

        print(phenotype.shape)
        print(phenotype)

        server.spawn_structure(phenotype, x, 4, 0)

        x += x_delta



def explore_hbrs(hbr_path1,
                hbr_config1,
                hbr_path2,
                hbr_config2,
                spawn_point:Tuple):
    """
    Interactive tool to explore two HBRs 
    
    """
    
    logging.info('Loading HBR...')
    r1, G1 = load(hbr_path1, hbr_config1)
    r2, G2 = load(hbr_path2, hbr_config2)

    d1 = r1.descriptors.at[r1.fitnesses != 1].set(jnp.inf)
    d2 = r2.descriptors.at[r2.fitnesses != 1].set(jnp.inf)
    server = ClientHandler() # minecraft server 
    
    input_desc = [
        ('BD 0/1 - start pos (x, y): ', tuple, 0, 19),
        ('BD 2/3 - end pos (x, y): ', tuple, 0, 19),
        ('BD 4 - traps_ratio: ', float, 0, 1),
        ('BD 5 - enemies_ratio: ', float, 0, 1)

    ]

    while True:
        
        X = input_bd(input_desc)
    
        # need to convert first two bds:
        start_pos = jnp.array(X.pop(0)) / 20
        end_pos = jnp.array(X.pop(0)) / 20

        target_bd = jnp.concatenate([start_pos, end_pos, jnp.array(X)])
        target_bd = jnp.expand_dims(target_bd, 0)

        # find closest value to bd_space
        closest_idx_1 = me_repertoire.get_cells_indices(target_bd, d1)
        closest_idx_2 = me_repertoire.get_cells_indices(target_bd, d2)

        found_bd1 = tree_map(lambda x: x[closest_idx_1], d1)
        found_bd2 = tree_map(lambda x: x[closest_idx_2], d2)

        genotype1 = tree_map(lambda x: x[closest_idx_1], r1.genotypes)
        level1 = G1.express(genotype1)[0]

        genotype2 = tree_map(lambda x: lax.collapse(x[closest_idx_2], 0, 2), r2.genotypes)
        level2 = G2.express(genotype2)[0]

        logging.info(f'Hierarchical Repertoire:')
        logging.info(f'Nearest BD: {found_bd1} at position {closest_idx_1}')
        logging.info(f'Fitness: {r1.fitnesses[closest_idx_1]}')
        logging.info(f'Distance: {jnp.sum(jnp.square(jnp.subtract(target_bd, found_bd1)))}')
        logging.info(f'Level:\n{level1}')


        logging.info(f'Flat Repertoire:')
        logging.info(f'Nearest BD: {found_bd2} at position {closest_idx_2}')
        logging.info(f'Fitness: {r2.fitnesses[closest_idx_2]}')
        logging.info(f'Distance: {jnp.sum(jnp.square(jnp.subtract(target_bd, found_bd2)))}')
        logging.info(f'Level:\n{level2}')


        clear_area(server, (spawn_point[0], 200), (4, 10), (spawn_point[1], 200))
        server.spawn_structure(level1, x_offset=spawn_point[0], z_offset=spawn_point[1], y_offset=4)
        server.spawn_structure(level2, x_offset=spawn_point[0], z_offset=spawn_point[1]+50, y_offset=4)

def explore_hbr(hbr_path,
                hbr_config):
    """
    Interactive tool to explore HBR 
    
    """

    logging.info('Loading HBR...')
    repertoire, G = load(hbr_path, hbr_config)
    descriptors = repertoire.descriptors.at[jnp.any(jnp.isnan(repertoire.descriptors), axis=(1))].set(jnp.inf)
    server = ClientHandler() # minecraft server 
    
    input_desc = [
        ('BD 0/1 - start pos (x, y): ', tuple, 0, 19),
        ('BD 2/3 - end pos (x, y): ', tuple, 0, 19),
        ('BD 4 - traps_ratio: ', float, 0, 1),
        ('BD 5 - enemies_ratio: ', float, 0, 1)

    ]

    while True:
        
        X = input_bd(input_desc)
    
        # need to convert first two bds:
        start_pos = jnp.array(X.pop(0)) / 20
        end_pos = jnp.array(X.pop(0)) / 20

        target_bd = jnp.concatenate([start_pos, end_pos, jnp.array(X)])
        target_bd = jnp.expand_dims(target_bd, 0)

        # find closest value to bd_space
        closest_idx = me_repertoire.get_cells_indices(target_bd, descriptors)
        found_bd = tree_map(lambda x: x[closest_idx], descriptors)

        logging.info(f'Nearest BD: {found_bd} at position {closest_idx}')
        logging.info(f'Fitness: {repertoire.fitnesses[closest_idx]}')
        logging.info(f'Distance: {jnp.sum(jnp.square(jnp.subtract(target_bd, found_bd)))}')

        # genotype = tree_map(lambda x: jnp.collapse(x[closest_idx], 0), repertoire.genotypes)
        genotype = tree_map(lambda x: x[closest_idx], repertoire.genotypes)
        # Function to print the shape and dtype of each leaf node
 
        level = G.express(genotype)[0]
        logging.info(f'Level:\n{level}')

        clear_area(server, (0, 20), (4, 10), (0, 20))
        t0 = time.time()
        server.spawn_structure(level, y_offset=5)
        logging.info(f'Spawned in {time.time() - t0}s!')

def explore_structure(structure_path,
                      structure_config):
    
    logging.info('Loading repertoire...')
    repertoire, G = load(structure_path, structure_config)
    centroids = repertoire.centroids.at[jnp.any(jnp.isnan(repertoire.descriptors), axis=(1))].set(jnp.inf)
    server = ClientHandler() # minecraft server 

    input_desc = [
        ('BD 0/1 - start pos (x, y): ', tuple, 0, 19),
        ('BD 2/3 - end pos (x, y): ', tuple, 0, 19),
        ('BD 4 - chamber connectivity: ', float, 0, 1)
    ]

    while True:

        X = input_bd(input_desc)

        # need to convert first two bds:
        start_pos = jnp.array(X.pop(0)) / 20
        end_pos = jnp.array(X.pop(0)) / 20

        target_bd = jnp.concatenate([start_pos, end_pos, jnp.array(X)])
        target_bd = jnp.expand_dims(target_bd, 0)
  
        # find closest value to bd_space
        closest_idx = me_repertoire.get_cells_indices(target_bd, centroids)[0]
        found_bd = tree_map(lambda x: x[closest_idx], repertoire.descriptors)

        logging.info(f'Nearest BD: {found_bd} at position {closest_idx}')
        logging.info(f'Fitness: {repertoire.fitnesses[closest_idx]}')
        logging.info(f'Distance: {jnp.sum(jnp.square(jnp.subtract(target_bd, found_bd)))}')

        genotype = tree_map(lambda x: x[closest_idx], repertoire.genotypes)
        level = G.express(genotype)[0]
        logging.info(f'Level:\n{level}')

        clear_area(server, (0, 20), (4, 10), (0, 20))
        t0 = time.time()
        server.spawn_structure(level, y_offset=5)
        logging.info(f'Spawned in {time.time() - t0}s!')


def hbr_spawn_random(path, config, target_fitness, num_samples, start_x=0, start_z=0):


    COL_SPACE = 30
    ROW_SPACE = 30
    NUM_COLS = 5
    ROW_COUNT = 0

    repertoire, G = load(path, config)
    idxes = [i for i, f in enumerate(repertoire.fitnesses) if f == target_fitness]
    samples = random.sample(idxes, num_samples)
    num_rows = num_samples // NUM_COLS + (num_samples % NUM_COLS > 0)

    server = ClientHandler()

    #clear_area(server, (0, 350), (4, 8), (0, 350))
    clear_area(server, (start_x, start_x+NUM_COLS*COL_SPACE), (4, 8), (start_z, start_z + num_rows*ROW_SPACE))

    spawn_x = start_x
    spawn_z = start_z

    for idx in samples:

        genotype = tree_map(lambda x:x[idx], repertoire.genotypes)
        phenotype = G.express(genotype)[0]
        server.spawn_structure(phenotype, x_offset=spawn_x, z_offset=spawn_z, y_offset=4)

        ROW_COUNT += 1

        if ROW_COUNT == NUM_COLS:

            spawn_x = start_x
            spawn_z += ROW_SPACE
            ROW_COUNT = 0

        else:
            spawn_x += COL_SPACE


def explore_bd(path:str,
               config:str,
               bd_axis:int, # axis to explore
               other_values:Array, # fixed values for other axes  
               n_spawn:int=10,
               x:int=0,
               y:int=4,
               z:int=0,
               x_delta:int=30,
               z_delta:int=0):
    """
    Spawn all genotypes moving along an axes in the BD space
    
    """
    logging.info('Loading repertoire...')

    repertoire, G = load(path, config)

    assert  bd_axis < repertoire.descriptors.shape[1] , 'bd_axes not in range!'

    bd_range = jnp.linspace(0, 1, n_spawn)

    other_values = jnp.hstack((other_values, other_values[0]))
    repertoire_descriptors =  repertoire.descriptors.at[jnp.any(jnp.isnan(repertoire.descriptors), axis=(1))].set(jnp.inf)
    descriptors = jnp.tile(other_values, (n_spawn, 1))
    descriptors = descriptors.at[:,bd_axis].set(bd_range)
    real_descriptors = me_repertoire.get_cells_indices(descriptors, repertoire_descriptors)

    template = jnp.full((20, 20), -1)
    server = ClientHandler() # minecraft server 
    clear_area(server, x_range=(x, (n_spawn*x_delta)*50), y_range=(4,8), z_range=(z, (z+(n_spawn*z_delta))+20))

    logging.info('Spawning....')


    for i in real_descriptors:
        genotype = tree_map(lambda x: x[i], repertoire.genotypes)
        bd = repertoire.descriptors[i]
        fitness = repertoire.fitnesses[i]
        logging.info(f'BD: {bd}')
        logging.info(f'Fitness: {fitness}')
        logging.info(f'Idx: {i}')
        phenotype = G.express(genotype)
        phenotype = lax.collapse(phenotype, 0 , 2)
        logging.info(f'Phenotype:\n {phenotype}')
        server.spawn_structure(phenotype, x, y+1, z)

        x += x_delta
        z += z_delta

def input_bd(input_desc):
    while True:
        inputs = []
        for desc, expected_type, min_val, max_val in input_desc:
            user_input = input(desc)
            
            if expected_type == tuple:
                try:
                    # Convert input into a tuple of floats
                    values = tuple(float(x) for x in user_input.split(','))
                    
                    # Ensure the tuple has exactly 2 elements
                    if len(values) != 2:
                        raise ValueError("Expected exactly two values.")
                    
                    # Check if all values are within the specified range
                    if not all(min_val <= v <= max_val for v in values):
                        raise ValueError(f"All tuple values must be between {min_val} and {max_val}.")
                    
                    inputs.append(values)
                    
                except ValueError as e:
                    print(f"Error: {e}")
                    break  # Restart the input process if any errors occur
                
            elif expected_type == float:
                try:
                    # Convert input to a float
                    value = float(user_input)
                    
                    # Check if the value is within the specified range
                    if not (min_val <= value <= max_val):
                        raise ValueError(f"Value must be between {min_val} and {max_val}.")
                    
                    inputs.append(value)
                    
                except ValueError as e:
                    print(f"Error: {e}")
                    break  # Restart the input process if any errors occur

        else:
            # Only return if all inputs are correctly parsed
            return inputs

def clear_area(server:ClientHandler, x_range:Tuple, y_range:Tuple, z_range:Tuple):

    logging.info("Clearing area....")
    t0 = time.time()

    server.fill_cube((x_range[0], y_range[0], z_range[0]), (x_range[1], y_range[1], z_range[1]), block_type=AIR) # reset surroundings
    logging.info(f"Cleared area in {time.time() - t0}s!")
