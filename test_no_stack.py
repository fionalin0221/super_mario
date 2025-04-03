import gym
import torch
from collections import deque
from train import DQNSolver
import numpy as np
from torchvision import transforms
import cv2

# transform = transforms.Compose([
#     transforms.ToPILImage(),                                   # Convert NumPy to PIL Image
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((90, 84)),
#     transforms.CenterCrop(84),
#     transforms.ToTensor(),                                     # Convert to tensor and normalize to [0, 1]
# ])

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.frames = deque([], maxlen=4)

        self.model = torch.jit.load('policy_model_best.pth').to(device)
        self.model.eval()  # Set to evaluation mode
        self.prev_obs = None
        # pass

    def act(self, observation):
        # if self.prev_obs:
        #     obs_np = np.array(observation)
        #     prev_obs_np = np.array(self.prev_obs)
        #     print(np.linalg.norm(prev_obs_np - obs_np))
        # self.prev_obs = observation
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        # frame = frame[3:87, :]
        frame = np.expand_dims(frame, axis=0)

        # self.frames.append(frame)
        # while len(self.frames) < 4:
        #     self.frames.append(frame)

        # print(f"Initial frames stacked shape: {[frame.shape for frame in self.frames]}")  # Debugging

        # input = np.concatenate(list(self.frames), axis=0)
        input = frame
        input = torch.tensor(input, dtype=torch.float32).unsqueeze(0) / 255.0
        # print(input.shape)
        
        action_prob = self.model(input.to(device))
        # print(action_prob)
        return torch.argmax(action_prob).item()
        # return self.action_space.sample()