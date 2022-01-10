import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='GridDriving-ray-v0',
    entry_point='gym_grid_driving_ray.envs:GridDrivingEnv'
)