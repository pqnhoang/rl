import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class CriticNet(nn.Module):
  def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128, name = 'critic', checkpoints_dir='tmp/td3', lr = 10e-3):
    super(CriticNet, self).__init__()
    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.n_actions = n_actions
    self.name = name
    self.checkpoints_dir = checkpoints_dir
    self.checkpoint_file = os.path.join(self.checkpoints_dir, name + '_td3')
    
    
    self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.q1 = nn.Linear(fc2_dims, 1)
    
    self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.005)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Create CriticNet using device: {self.device}")
    self.to(self.device)
    
  def forward(self, state, action):
    action_value = self.fc1(torch.cat([state, action], dim=1))
    action_value = F.relu(action_value)
    action_value = self.fc2(action_value)
    action_value = F.relu(action_value)
    
    q1 = self.q1(action_value)
    return q1
  
  def save_checkpoint(self):
    torch.save(self.state_dict(), self.checkpoint_file)
  
  def load_checkpoint(self):
    self.load_state_dict(torch.load(self.checkpoint_file))

class ActorNet(nn.Module):
  def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, lr = 10e-3, n_actions = 2, name = 'actor', checkpoints_dir='tmp/td3'):
    super(ActorNet, self).__init__()
    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.name = name
    self.checkpoints_dir = checkpoints_dir
    self.checkpoint_file = os.path.join(self.checkpoints_dir, name + '_td3')
    
    self.fc1 = nn.Linear(*self.input_dims, fc1_dims)
    self.fc2 = nn.Linear(fc1_dims, fc2_dims)
    self.output = nn.Linear(fc2_dims, n_actions)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    print(f"Created Actor Using device: {self.device}")
    self.to(self.device)
    
  def forward(self, state):
    x = self.fc1(state)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    
    x = torch.tanh(self.output(x))
    return x
  
  def save_checkpoint(self):
    torch.save(self.state_dict(), self.checkpoint_file)
  
  def load_checkpoint(self):
    self.load_state_dict(torch.load(self.checkpoint_file))