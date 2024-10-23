import sys
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60"

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
import functools
import matplotlib.pyplot as plt

from jax.lib import xla_bridge
from jax import vmap
from jax.flatten_util import ravel_pytree
from functools import partial
from gameplay.genotype import GameplayGenotype
from structure.genotype import StructureGenotype
from common.utils import load_genotype, get_scoring_fn
from hierarchy.scoring_fns import BasicScore

from omegaconf import DictConfig, OmegaConf
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, compute_euclidean_centroids, compute_cvt_centroids
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_multidimensional_map_elites_grid, plot_2d_map_elites_repertoire


@hydra.main(config_path="configs", config_name="map_elites.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.info(cfg)
    logging.info("Training")
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(config=wandb_cfg, **cfg.wandb):

        # Create local results directory for visualizations and repertoire
        results_dir = os.path.join(os.getcwd(), cfg.results_dir)
        _repertoire_dir = os.path.join(results_dir, "repertoire", "")
        os.makedirs(_repertoire_dir, exist_ok=True)

        key = jrand.PRNGKey(cfg.seed)

        G = load_genotype(cfg.tasks['name'], 
                          **cfg.tasks.genotype_setup, 
                          batch_size=cfg.batch_size, 
                          logger=logging)
        
        scoring_fn = get_scoring_fn(cfg.tasks['name'], cfg.tasks.scoring_fn)
        G.set_scoring_fn(scoring_fn, **cfg.tasks.scoring_fn_args)

        init_genotypes, key = G.generate_genotypes(key)

        mixing_emitter = MixingEmitter(
            mutation_fn = G.mutation_fn,
            variation_fn = G.variation_fn,
            variation_percentage=cfg.variation_p, 
            batch_size = cfg.batch_size
        )

        metrics_fn = partial(default_qd_metrics, qd_offset=0)

        logging.info("Algorithm initialization")
        init_time = time.time()

        map_elites = MAPElites(
            scoring_function=G.score_genotypes,
            emitter=mixing_emitter,
            metrics_function=metrics_fn
        )

        centroids = compute_euclidean_centroids(cfg.tasks.grid_shape, 
                                                minval=cfg.tasks.bd_min, 
                                                maxval=cfg.tasks.bd_max)


        # Compute initial repertoire and emitter state

        repertoire, emitter_state, key = map_elites.init(init_genotypes, centroids, key)

        init_repertoire_time = time.time() - init_time
        logging.info(f"Repertoire initialized in {init_repertoire_time:.2f} seconds")


        num_loops = int(cfg.num_iterations / cfg.log_period)
        total_evaluations = 0
        plot_errors = 0
        all_metrics = {}

        map_elites_scan_update = map_elites.scan_update

        for i in range(num_loops):

            start_time = time.time()
            (repertoire, emitter_state, key,), metrics = jax.lax.scan(
                map_elites_scan_update,
                (repertoire, emitter_state, key),
                (),
                length=cfg.log_period,
            )

            timelapse = time.time() - start_time
            logging.info(f"Loop {i + 1}/{num_loops} - Time: {timelapse:.2f} seconds")

            metrics_last_iter = jax.tree_util.tree_map(
                lambda metric: "{0:.2f}".format(float(metric[-1])), metrics
            )
            logging.info(metrics_last_iter)

            # log metrics
            total_evaluations += cfg.log_period * cfg.batch_size

            logged_metrics = {
                "time": timelapse,
                "evaluations": total_evaluations,
                "iteration": 1 + i * cfg.log_period,
            }

            for k, v in metrics.items():
                # take last value
                logged_metrics[k] = v[-1]
                # take all values
                if k in all_metrics.keys():
                    all_metrics[k] = jnp.concatenate([all_metrics[k], v])
                else:
                    all_metrics[k] = v

            wandb.log(logged_metrics)

            if (i + 1) % cfg.save_period == 0:
                repertoire.save(path=_repertoire_dir)
                logging.info(f"Repertoire saved at iteration {i + 1} to {_repertoire_dir}")

            if (i + 1) % cfg.img_period == 0:

                try:
                
                    if len(cfg.tasks.bd_min) > 2: 
                        pass
                        fig, ax = plot_multidimensional_map_elites_grid(repertoire,
                                                                        min(cfg.tasks.bd_min), 
                                                                        max(cfg.tasks.bd_max),
                                                                        tuple(cfg.tasks.grid_shape))
                    else:
                        fig, ax = plot_2d_map_elites_repertoire(repertoire.centroids,
                                                                repertoire.fitnesses,
                                                                min(cfg.tasks.bd_min),
                                                                max(cfg.tasks.bd_max))
                    
                    wandb.log({"map_elites_grid": wandb.Image(fig)})
                    fig.savefig(os.path.join(results_dir, f'plot_{i+1}.png'))

                except Exception as e:
                    logging.debug("WARNING! PLOT ERROR.")
                    plot_errors += 1
                    wandb.log({"plot errors": {plot_errors}})


        
        total_duration = time.time() - init_time

        wandb.log({"final/duration": total_duration})
        repertoire.save(path=_repertoire_dir)
        
        for k, v in metrics.items():
            wandb.log({"final/" + k: v[-1]})

        try:

            if len(cfg.tasks.bd_min) > 2: 
                
                fig, ax = plot_multidimensional_map_elites_grid(repertoire,
                                                                min(cfg.tasks.bd_min), 
                                                                max(cfg.tasks.bd_max),
                                                                tuple(cfg.tasks.grid_shape))
            else:
                fig, ax = plot_2d_map_elites_repertoire(repertoire.centroids,
                                                        repertoire.fitnesses,
                                                        min(cfg.tasks.bd_min),
                                                        max(cfg.tasks.bd_max))
            
            wandb.log({"map_elites_grid": wandb.Image(fig)})
            fig.savefig(os.path.join(results_dir, 'plot.png'))

        except Exception as e:
                logging.debug("WARNING! PLOT ERROR.")
                plot_errors += 1
                wandb.log({"plot errors": {plot_errors}})

 

if __name__ == "__main__":
    main()
