import gym
import gym_grid_driving_ray
from gym_grid_driving_ray.envs.grid_driving import LaneSpec, MaskSpec, Point, GridDrivingEnv
import numpy as np

import argparse

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from ray.rllib.models.catalog import ModelCatalog

def ray_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=6, type=int)
    parser.add_argument("--training-iteration", default=10000, type=int)
    parser.add_argument("--ray-num-cpus", default=7, type=int)
    args = parser.parse_args()
    ray.init(num_cpus=args.ray_num_cpus, local_mode=True)

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    # Registering the custom env
    register_env("drivingenv", lambda config: GridDrivingEnv(config))

    tune.run("contrib/AlphaZero", stop={"training_iteration": args.training_iteration}, 
    config=     {    
                        "env": "drivingenv",
                        "env_config": {
                                "lanes": [LaneSpec(1, [-1, -1]), LaneSpec(1, [-1, -1]), LaneSpec(1, [-1, -1])], 
                                "width": 8, "agent_speed_range": (-1,-1), "finish_position": Point(0,1), 
                                "agent_pos_init": Point(6,1), "stochasticity": 1.0, "tensor_state": False, 
                                "flicker_rate": 0.5, "mask": MaskSpec('follow', 1), "random_seed": 13, "observation_type": 'vector'
                        }, 
                        # "num_workers": args.num_workers,
                        # "rollout_fragment_length": 50,
                        # "tra]in_batch_size": 500,
                        # "sgd_minibatch_size": 64,
                        # "lr": 1e-4,
                        # "num_sgd_iter": 1,
                        # "mcts_config":  {
                        # "puct_coefficient": 1.5,
                        # "num_simulations": 100,
                        # "temperature": 1.0,
                        # "dirichlet_epsilon": 0.20,
                        # "dirichlet_noise": 0.03,
                        # "argmax_tree_policy": False,
                        # "add_dirichlet_noise": True,
                                        # }
                },       
        #     max_failures=0, 
        #     checkpoint_at_end=True, 
        #     checkpoint_freq=1
        )

if __name__ == "__main__":
    ray_main()