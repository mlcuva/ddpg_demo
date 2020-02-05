import os

import numpy as np
import torch

import nets
import utils

class Agent(torch.nn.Module):
    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
    
    def eval(self):
        self.actor.eval()
        self.critic.eval()
    
    def train(self):
        self.actor.train()
        self.critic.train()
    
    def save(self, path):
        actor_path = os.path.join(path, 'actor.pt')
        critic_path = os.path.join(path, 'critic.pt')
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    def load(self, path):
        actor_path = os.path.join(path, 'actor.pt')
        critic_path = os.path.join(path, 'critic.pt')
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

    def forward(self, state):
        # first need to add batch dimension and convert to torch tensors
        state = np.expand_dims(state, 0).astype(np.float32)
        state = torch.from_numpy(state)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        return np.squeeze(action.cpu(), 0)
 
class MountaincarAgent(Agent):
    def __init__(self):
        super().__init__()
        self.actor = nets.MountaincarActor()
        self.critic = nets.MountaincarCritic()

class PendulumAgent(Agent):
    def __init__(self):
        super().__init__()
        self.actor = nets.PendulumActor()
        self.critic = nets.PendulumCritic()