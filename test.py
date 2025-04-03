import gym
import torch
import torch.nn as nn
from collections import deque
import numpy as np
# import cv2
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DQNSolver(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.frames = deque([], maxlen=4)

        self.model = torch.jit.load('policy_model_latest.pth').to(device)
        self.model.eval()  # Set to evaluation mode
        self.prev_action = 0
        self.count = 0

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((84, 84))
        ])


    def act(self, observation):
        # select action only at no skpping frames        
        if self.count % 4 == 0:
            # convert frame to gray scale
            # obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0) / 255
            # frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            # resize the frame
            # frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
            # frame = frame[3:87, :]
            # frame = np.expand_dims(frame, axis=0)

            obs = torch.tensor(observation.copy()).permute(2, 0, 1)
            frame = self.transform(obs)
            frame = np.array(frame)

            self.frames.append(frame)
            while len(self.frames) < 4:
                self.frames.append(frame)

            input = np.concatenate(list(self.frames), axis=0)
            input = torch.tensor(input, dtype=torch.float32).unsqueeze(0) / 255.0
        
            action_values = self.model(input.to(device))
            action = torch.argmax(action_values).item()
            self.prev_action = action
        else:
            action = self.prev_action

        self.count += 1

        return action