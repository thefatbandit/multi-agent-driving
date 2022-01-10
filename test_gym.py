import gym
import gym_grid_driving_gym
from gym_grid_driving_gym.envs.grid_driving import LaneSpec, MaskSpec, Point, GridDrivingEnv
import numpy as np

import argparse

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from ray.rllib.models.catalog import ModelCatalog

lanes = [
        LaneSpec(2, [-1, -1]),
        LaneSpec(2, [-2, -1]),
        LaneSpec(3, [-3, -1]),
        ]

def run_one_episode (env, verbose=False):
    env.reset()
    sum_reward = 0

    for i in range(500):
        action = env.action_space.sample()

        if verbose:
            print("action:", action)

        state, reward, done, info = env.step(action)
        sum_reward += reward

        if verbose:
            env.render()

        if done:
            if verbose:
                print("done @ step {}".format(i))

            break

    if verbose:
        print("cumulative reward", sum_reward)

    return sum_reward

def gym_main():
    # first, create the custom environment and run it for one episode
    env = gym.make('GridDriving-gym-v0', lanes=lanes, width=8, 
                agent_speed_range=(-3,-1), finish_position=Point(0,1), agent_pos_init=Point(6,1),
                stochasticity=1.0, tensor_state=False, flicker_rate=0.5, mask=MaskSpec('follow', 2), random_seed=13)

    sum_reward = run_one_episode(env, verbose=True)

    # next, calculate a baseline of rewards based on random actions (no policy)
    history = []

    for _ in range(10000):
        sum_reward = run_one_episode(env, verbose=False)
        history.append(sum_reward)

    avg_sum_reward = sum(history) / len(history)
    print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))

if __name__ == "__main__":
    gym_main()  