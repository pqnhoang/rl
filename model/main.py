import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config
import time


from model.model import ActorNet, CriticNet
from model.buffer import ReplayBuffer
from model.td3 import Agent

if __name__ == "__main__":
  if not os.path.exists('tmp/td3'):
    os.makedirs('tmp/td3')
  
  env_name = "Door"
  
  config = load_controller_config(default_controller="JOINT_VELOCITY")
  env = suite.make(
    env_name=env_name,
    robots=["Panda"],
    controller_configs=config,
    has_renderer=False,
    use_camera_obs=False,
    horizon=300,
    reward_shaping=True,
    control_freq=20,
    )
  env = GymWrapper(env)
  
  
  actor_learning_rate = 0.001
  critic_learning_rate = 0.001
  batch_size = 128
  layer_1_size = 256
  layer_2_size = 128
  
  agent = Agent(alpha=actor_learning_rate,
                beta=critic_learning_rate,
                tau=0.005,
                input_dims=env.observation_space.shape,
                env=env,
                n_actions=env.action_space.shape[0],
                layer1_size=layer_1_size,
                layer2_size=layer_2_size,
                batch_size=batch_size)
  
  writer = SummaryWriter(log_dir='runs/logs')
  n_games = 10000
  best_score = 0
  episode_identifier = f"0 - actor_learning_rate = {actor_learning_rate} critic_learning_rate = {critic_learning_rate} batch_size = {batch_size} layer_1_size ={layer_1_size} layer_2_size = {layer_2_size}"
  
  agent.load_models()
  
  for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    
    while not done:
      action = agent.choose_action(observation)
      next_observation, reward, done, info = env.step(action)
      score += reward
      agent.remember(observation, action, reward, next_observation, done)
      agent.learn()
    writer.add_scalar(f'Score/{episode_identifier}', score, global_step = i)
    
    if (i%10 == 0):
      agent.save_models()
    
    print(f'Episode {i} Score {score}')