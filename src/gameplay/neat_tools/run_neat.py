"""
Script for running NEAT experiments for generating gameplay operators 

"""
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import hydra
import jax.random as jrand
import time
import jax
import jax.numpy as jnp 
import wandb
import hydra
import yaml
import numpy as np
import argparse
import logging
import gameplay.fitnesses
import gameplay.problem
import os

from jax.nn import softmax
from omegaconf import DictConfig, OmegaConf
from tensorneat.pipeline import Pipeline
from tensorneat import algorithm, genome, problem, common

from gameplay.problem import GameplayProblem
from gameplay.fitnesses import GameplayFitnessFunction, load_fitness_fn




@hydra.main(config_path="configs", config_name="neat.yaml", version_base=None)
def main(cfg:DictConfig):
    
    print('CWD:', os.getcwd())



    logging.info(cfg)
    logging.info("Training")

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(config=wandb_cfg, **cfg.wandb):


        neat = algorithm.NEAT(
            pop_size=cfg.pop_size,
            species_size=cfg.species_size,
            survival_threshold=cfg.survival_threshold,

            genome=genome.DefaultGenome(
                num_inputs=cfg.num_inputs,
                num_outputs=cfg.num_outputs,
                output_transform=softmax,
                max_conns=cfg.max_connections 
            ),
        )

        GameplayFitnessFunction.n_placeable_features = cfg.num_outputs - 1
        fitness_fn = load_fitness_fn(cfg.tasks)
    
        logging.info("Setting up problem")

        task = GameplayProblem(
            fitness_fn=fitness_fn,
            n_tiles=cfg.n_tiles,
            window_size=cfg.window_size,
            n_levels=cfg.n_levels,
            RNGKey=jrand.PRNGKey(cfg.seed),
            repertoire_path=cfg.repertoire_path,
            structure_config=cfg.structure_path,
            logger = logging
        )

        pipeline = Pipeline(
            neat,
            task,
            generation_limit=cfg.generation_limit,
            fitness_target=-cfg.fitness_target,
            seed=cfg.seed,
        )

        logging.info("Algorithm initialization")

        state = pipeline.setup()
        logging.info("Running NEAT!")

        # run until termination 
        state, best = pipeline.auto_run(state)
        # show best
        pipeline.show(state, best, logger=logging)
        # save best 







if __name__ == "__main__":
    main()

