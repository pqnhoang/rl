import os, warnings
import time
warnings.filterwarnings("ignore")

import gym
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper

from torch.utils.tensorboard import SummaryWriter

LOGDIR = "runs/door_sac"
os.makedirs(LOGDIR, exist_ok=True)

def make_env(seed=0):
  def _fn():
    config = load_controller_config(default_controller="JOINT_VELOCITY")
    env = suite.make(
      env_name="Door",
      robots="Panda",
      controller_configs=config,
      has_renderer=False,
      use_camera_obs=False,
      horizon=300,
      reward_shaping=True,
      control_freq=20,
      )
    env = GymWrapper(env)
    env.seed(seed)
    return env
  return _fn

# env_fn = make_env()
# env = env_fn()
# print(env.action_space.high)
# print(env.action_space.low)
# print(env.observation_space.shape)
# for _ in range(10):
#   obs = env.reset()
#   print(obs)
#   for _ in range(10):
#     action = env.action_space.sample()
#     print(action)
#     obs, reward, done, info = env.step(action)
#     print(obs, reward, done, info)
#     if done:
#       break
# env.close()

