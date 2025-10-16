import os
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.distributions import Normal

class CriticNet(nn.Module):
  def __init__(self, state_dim, action_dim, fc1_dim=256, fc2_dim=128, name = 'critic', checkpoints_dir='tmp/', lr = 10e-3):
    super(CriticNet, self).__init__()
    self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
    self.fc2 = nn.Linear(fc1_dim, fc2_dim)
    self.q1 = nn.Linear(fc2_dim, 1)
    
    self.name = name
    self.checkpoints = checkpoints_dir
    self.checkpoint_file = os.path.join(self.checkpoints, name)
    self.lr = lr
    self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
    
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {self.device}")
    self.to(self.device)
    
  def forward(self, state, action):
    x = torch.cat([state, action], dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.q1(x)
  
  def save_checkpoint(self):
    torch.save(self.state_dict(), self.checkpoint_file)
    print(f"Checkpoint saved to {self.checkpoint_file}")
  
  def load_checkpoint(self):
    self.load_state_dict(torch.load(self.checkpoint_file))
    print(f"Checkpoint loaded from {self.checkpoint_file}")


class ActorNet(nn.Module):
  def __init__(self, state_dim, action_dim, fc1_dim=256, fc2_dim=128, name = 'actor', checkpoints_dir='tmp/', lr = 10e-3):
    super(ActorNet, self).__init__()
    self.fc1 = nn.Linear(state_dim, fc1_dim)
    self.fc2 = nn.Linear(fc1_dim, fc2_dim)
    self.output = nn.Linear(fc2_dim, action_dim)
    
    self.name = name
    self.checkpoints = checkpoints_dir
    self.checkpoint_file = os.path.join(self.checkpoints, name)
    self.lr = lr
    self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {self.device}")
    self.to(self.device)
    
  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    x = F.tanh(self.output(x))
    return x
  
  def save_checkpoint(self):
    torch.save(self.state_dict(), self.checkpoint_file)
    print(f"Checkpoint saved to {self.checkpoint_file}")
  
  def load_checkpoint(self):
    self.load_state_dict(torch.load(self.checkpoint_file))
    print(f"Checkpoint loaded from {self.checkpoint_file}")