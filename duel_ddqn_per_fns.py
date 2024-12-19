import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import time
import csv

np.random.seed(100)
torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

def save_to_csv(values, filename="training.csv"):
    with open(filename, 'a') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(values)
        f_object.close()

# Define the DQN architecture
class DQN(nn.Module):
    def __init__(self, input_shape, action_space):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(np.prod(input_shape), 256)
        
        # Dueling DQN specific layers
        self.fc_value = nn.Linear(256, 1)
        self.fc_advantage = nn.Linear(256, action_space)
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        
        # Combine value and advantage to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


def get_epsilon(it, max_epsilon=1.0, min_epsilon=0.01, decay=500):
    epsilon= max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * it / decay)
    return epsilon

def choose_action(state, policy_net, epsilon):
    flag=0
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32, device=device)
            q_values = policy_net(state)
            action = q_values.max(1)[1].item()
            flag=1
    else:
        action = env.action_space.sample()

    return action,flag


environment= "LunarLander-v2"
env = gym.make(environment)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
