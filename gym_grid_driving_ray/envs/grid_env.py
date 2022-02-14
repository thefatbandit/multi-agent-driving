from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box

import gym_grid_driving_ray
from gym_grid_driving_ray.envs.grid_driving import LaneSpec, MaskSpec, Point, GridDrivingEnv

lanes = [
        LaneSpec(1, [-1, -1]),
        LaneSpec(1, [-1, -1]),
        LaneSpec(1, [-1, -1]),
        ]

class Grid:
    """
    Wrapper for the gym Grid Driving Env where the reward is accumulated to the end
    """

    def __init__(self, config=None):
        self.env = gym.make('GridDriving-ray-v0')
        self.action_space = Discrete(3)
        self.observation_space = Dict(
            {
                "obs": self.env.observation_space,
                "action_mask": Box(low=0, high=1, shape=(self.action_space.n,)),
            }
        )
        self.running_reward = 0

    def reset(self):
        self.running_reward = 0
        return {
            "obs": self.env.reset(),
            "action_mask": np.array([1]*self.action_space.n, dtype=np.float32),
        }
    def step(self,action):
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0
        return (
            {"obs": obs, "action_mask": np.array([1]*self.action_space.n, dtype=np.float32)}, 
            score, 
            done, 
            info,
        )
    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))
        return {"obs": obs, "action_mask": np.array([1]*self.action_space.n, dtype=np.float32)}
    
    def get_state(self):
        return deepcopy(self.env), self.running_reward