import os
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
import torch.optim as optim
from torch.distributions import Normal
from model.model import ActorNet, CriticNet
from model.buffer import ReplayBuffer
import numpy as np



class Agent:
  def __init__(self, alpha, beta, state_dim, tau, env, gamma=0.99, update_actor_interval=2, warmup=1000,
                                 action_dim = 2, max_size=1000000, fc1_dim=256, fc2_dim=128, batch_size=100, lr=10e-3, noise=0.1):
    self.gamma = gamma
    self.tau = tau
    self.max_action = env.action_space.high
    self.min_action = env.action_space.low
    self.memory = ReplayBuffer(max_size, state_dim, action_dim)
    self.batch_size = batch_size
    self.learn_step_cntr = 0
    self.time_step = 0
    self.warmup = warmup
    self.n_actions = action_dim
    self.update_actor_inter = update_actor_interval
    
    # Networks
    self.actor = ActorNet(state_dim, action_dim, fc1_dim, fc2_dim,
                          name = 'actor', checkpoints_dir='tmp/', lr = alpha)
    self.critic_1 = CriticNet(state_dim, action_dim, fc1_dim, fc2_dim,
                              name = 'critic_1', checkpoints_dir='tmp/', lr = beta)
    self.critic_2 = CriticNet(state_dim, action_dim, fc1_dim, fc2_dim,
                              name = 'critic_2', checkpoints_dir='tmp/', lr = beta)
    
    # Target networks
    self.actor_target = ActorNet(state_dim, action_dim, fc1_dim, fc2_dim,
                                name = 'actor_target', checkpoints_dir='tmp/', lr = alpha)
    self.critic_1_target = CriticNet(state_dim, action_dim, fc1_dim, fc2_dim,
                                name = 'critic_1_target', checkpoints_dir='tmp/', lr = beta)
    self.critic_2_target = CriticNet(state_dim, action_dim, fc1_dim, fc2_dim,
                                name = 'critic_2_target', checkpoints_dir='tmp/', lr = beta)
    
    self.noise = noise
    self.update_network_parameters(tau=tau)
  
  def choose_action(self, observation, validation=False):
    if self.time_step < self.warmup and not validation:
      mu = torch.tensor(np.random.normal(0, self.noise, size=self.n_actions), dtype=torch.float32).to(self.actor.device)
    else:
      state = torch.tensor(observation, dtype=torch.float32).to(self.actor.device)
      mu = self.actor.forward(state).to(self.actor.device)
    
    mu_prime = mu + torch.tensor(np.random.normal(0, self.noise, size=self.n_actions), dtype=torch.float32).to(self.actor.device)
    mu_prime = mu_prime.clip(self.min_action[0], self.max_action[0])
    self.time_step += 1
    return mu_prime.detach().cpu().numpy()
  
  def remember(self,state,action,reward,next_state,done):
    self.memory.store_transition(state, action, reward, next_state, done)
    
  
  def learn(self):
    if self.memory.mem_cntr < self.batch_size * 10:
      return
    
    state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
    
    reward = torch.tensor(reward, dtype=torch.float32).to(self.critic_1.device)
    done = torch.tensor(done, dtype=torch.bool).to(self.critic_1.device)
    next_state = torch.tensor(next_state, dtype=torch.float32).to(self.critic_1.device)
    state = torch.tensor(state, dtype=torch.float32).to(self.critic_1.device)
    action = torch.tensor(action, dtype=torch.float32).to(self.critic_1.device)
    
    
    target_actions = self.actor_target.forward(next_state)
    target_actions = target_actions + torch.clamp(torch.tensor(np.random.normal(scale=0.2)), min= -0.5, max= 0.5)
    target_actions = torch.clamp(target_actions, self.min_action[0], self.max_action[0])
    
    next_q1 = self.critic_1_target.forward(next_state, target_actions)
    next_q2 = self.critic_2_target.forward(next_state, target_actions)
    
    q1 = self.critic_1.forward(state, action)
    q2 = self.critic_2.forward(state, action)
    
    next_q1[done] = 0.0
    next_q2[done] = 0.0
    
    next_q1 = next_q1.view(-1)
    next_q2 = next_q2.view(-1)
    
    
    q1 = self.critic_1.forward(state, action)
    q2 = self.critic_2.forward(state, action)
    
    next_critic_value = torch.min(next_q1, next_q2)
    
    target = reward + self.gamma * next_critic_value
    target = target.view(self.batch_size, -1)
    
    q1_loss = nn_functional.mse_loss(q1, target)
    q2_loss = nn_functional.mse_loss(q2, target)
    
    critic_loss = q1_loss + q2_loss
    critic_loss.backward()
    
    self.critic_1.optimizer.step()
    self.critic_2.optimizer.step()
    
    self.learn_step_cntr += 1
    
    if self.learn_step_cntr % self.update_actor_inter != 0:
      return
    
    self.actor.optimizer.zero_grad()
    actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
    actor_loss = -torch.mean(actor_q1_loss)
    actor_loss.backward()
    
    self.actor.optimizer.step()
    self.update_network_parameters()
    
    
  def update_network_parameters(self, tau=None):
    if not tau:
      tau = self.tau
      
      
    actor_params = self.actor.named_parameters()
    critic_1_params = self.critic_1.named_parameters()
    critic_2_params = self.critic_2.named_parameters()
    target_actor_params = self.actor_target.named_parameters()
    target_critic_1_params = self.critic_1_target.named_parameters()
    target_critic_2_params = self.critic_2_target.named_parameters()

    actor_state_dict = dict(actor_params)
    critic_1_state_dict = dict(critic_1_params)
    critic_2_state_dict = dict(critic_2_params)
    target_actor_state_dict = dict(target_actor_params)
    target_critic_1_state_dict = dict(target_critic_1_params)
    target_critic_2_state_dict = dict(target_critic_2_params)
    
    for name in actor_state_dict:
      actor_state_dict[name] = tau*actor_state_dict[name] + (1-tau)*target_actor_state_dict[name]
      
    for name in critic_1_state_dict:
      critic_1_state_dict[name] = tau*critic_1_state_dict[name] + (1-tau)*target_critic_1_state_dict[name]
      
    for name in critic_2_state_dict:
      critic_2_state_dict[name] = tau*critic_2_state_dict[name] + (1-tau)*target_critic_2_state_dict[name]
      
    self.actor_target.load_state_dict(actor_state_dict)
    self.critic_1_target.load_state_dict(critic_1_state_dict)
    self.critic_2_target.load_state_dict(critic_2_state_dict)
  def save_models(self):
    self.actor.save_checkpoint()
    self.critic_1.save_checkpoint()
    self.critic_2.save_checkpoint()
    self.actor_target.save_checkpoint()
    self.critic_1_target.save_checkpoint()
    self.critic_2_target.save_checkpoint()
    print('Models saved successfully')
    
  def load_models(self):
    success_count = 0
    total_models = 6
    
    if self.actor.load_checkpoint():
      success_count += 1
    if self.critic_1.load_checkpoint():
      success_count += 1
    if self.critic_2.load_checkpoint():
      success_count += 1
    if self.actor_target.load_checkpoint():
      success_count += 1
    if self.critic_1_target.load_checkpoint():
      success_count += 1
    if self.critic_2_target.load_checkpoint():
      success_count += 1
    
    if success_count == total_models:
      print('All models loaded successfully')
    elif success_count > 0:
      print(f'Warning: Only {success_count}/{total_models} models loaded successfully')
    else:
      print('No existing models found - starting fresh')
    