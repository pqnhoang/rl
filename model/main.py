import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import time


from model.env import make_env
from model.model import ActorNet, CriticNet
from model.buffer import ReplayBuffer
from model.td3 import Agent

if __name__ == "__main__":
  env = make_env()
  env = env()
  actor_learning_rate = 0.001
  critic_learning_rate = 0.001
  batch_size = 64
  layer_1_size = 256
  layer_2_size = 128
  
  agent = Agent(alpha=actor_learning_rate,
                beta=critic_learning_rate,
                state_dim=env.observation_space.shape[0],
                tau=0.005,
                env=env,
                action_dim=env.action_space.shape[0],
                fc1_dim=layer_1_size,
                fc2_dim=layer_2_size,
                batch_size=batch_size)
  
  writer = SummaryWriter(log_dir='runs/td3')
  n_games = 10000
  best_score = 0
  episode_identifier = f"0_{time.time()} actor_learning_rate_{actor_learning_rate} critic_learning_rate_{critic_learning_rate} batch_size_{batch_size} layer_1_size_{layer_1_size} layer_2_size_{layer_2_size}"
  
  for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
      action = agent.choose_action(observation)
      observation_, reward, done, info = env.step(action)
      score += reward
      agent.remember(observation, action, reward, observation_, done)
      agent.learn()
    writer.add_scalar(f'Score/{episode_identifier}', score, i)
    
    if (i%10):
      agent.save_models()
    
    print(f'Episode {i} Score {score}')