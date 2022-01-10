import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='GridDriving-gym-v0',
    entry_point='gym_grid_driving_gym.envs:GridDrivingEnv'
)