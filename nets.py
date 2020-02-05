
import torch
from torch import nn
import torch.nn.functional as F


class MountaincarActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        act = torch.tanh(self.out(x))
        return act

class MountaincarCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 400)
        self.fc2 = nn.Linear(401, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)
    
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat((x, action), dim=1)))
        x = F.relu(self.fc3(x))
        val = self.out(x)
        return val

class PendulumActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 400)
        self.fc3 = nn.Linear(400, 128)
        self.out = nn.Linear(128, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        act = torch.tanh(self.out(x))
        return act

class PendulumCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 400)
        self.fc2 = nn.Linear(401, 300)
        self.fc3 = nn.Linear(300, 128)
        self.out = nn.Linear(128, 1)
    
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x_act = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x_act))
        x = F.relu(self.fc3(x))
        val = self.out(x)
        return val