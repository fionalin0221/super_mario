import cv2
import gym
import pygame
import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from collections import deque, namedtuple
from tqdm import tqdm
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym_super_mario_bros

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class ProcessImage(gym.ObservationWrapper):
    def __init__(self, env, width = 84, height = 84):
        super(ProcessImage, self).__init__(env)
        self._width = width
        self._height = height

        original_space = self.observation_space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, self._height, self._width), dtype=np.uint8)
    
    def observation(self, obs):
        # print(obs.shape, type(obs))
        frame = ((obs[0]+obs[1]+obs[2]) / 3).astype(np.uint8)
        # frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 90), interpolation=cv2.INTER_AREA)
        crop_h = (90 - self._height) // 2
        frame = frame[crop_h:crop_h + self._height, :]
        frame = np.expand_dims(frame, axis=0)
        return frame
    
class FrameStack(gym.Wrapper):
    def __init__(self, env, num_frames=4):
        gym.Wrapper.__init__(self, env)
        self.num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)

        original_space = self.observation_space
        original_shape = self.observation_space.shape

        # Define the new shape with k stacked frames along the channel axis
        # new_shape = (num_frames, ) + original_shape[1:] # (4, 84, 84)

        # self.observation_space = gym.spaces.Box(
        #     low=env.observation_space.low.repeat(num_frames, axis=0),
        #     high=env.observation_space.high.repeat(num_frames, axis=0),
        #     shape = new_shape,
        #     dtype=env.observation_space.dtype,
        # )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        for _ in range(self.num_frames):
            self.frames.append(obs)
        # print(f"Initial frames stacked shape: {[frame.shape for frame in self.frames]}")  # [(1, 84, 84), (1, 84, 84), (1, 84, 84), (1, 84, 84)]
        return self._get_obs()
    
    def step(self, action):
        rewards = 0
        done = False
        for _ in range(self.num_frames):
            # next_obs, reward, terminanted, truncated, info  = self.env.step(action)
            next_obs, reward, done, info  = self.env.step(action)
            # print(f"Next frame shape: {next_obs.shape}")  #(1, 84, 84)
            self.frames.append(next_obs)  # Add the new frame to the stack
            rewards += reward
            if done:
                break

        return self._get_obs(), rewards, done, info
    
    def _get_obs(self):
        """Return the stacked frames as a single observation."""
        assert len(self.frames) == self.num_frames  # Make sure the deque has k frames
        # print(f"Stacked frames shape before concatenation: {[frame.shape for frame in self.frames]}")  # [(1, 84, 84), (1, 84, 84), (1, 84, 84), (1, 84, 84)]
        return np.concatenate(list(self.frames), axis=0)  # Stack the frames along the last axis


def make_env():
    # env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='rgb_array')
    env = gym_super_mario_bros.make('SuperMarioBros-v0') # (240, 256, 3) class <'numpy.ndarray'>

    env = ProcessImage(env)
    # obs = env.reset()

    # print("Observation shape:", obs.shape, type(obs)) #(1, 84, 84) <class 'numpy.ndarray'> 
    # cv2.imshow("Mario Observation", obs[0])
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    # for i in range(16):
    #     cv2.imshow("Mario Observation", obs)
    #     cv2.imwrite(f'input_{i}.png', obs)
    #     cv2.waitKey(1000)
    #     cv2.destroyAllWindows()
    #     obs, reward, done, info  = env.step(5)

    env = FrameStack(env)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    # obs = env.reset()
    # print("Observation shape after stack:",obs.shape, type(obs))

    # for j in range(100):
    #     for i in range(len(obs)):
    #         cv2.imshow("Mario Observation", obs[i])
    #         # cv2.imwrite(f'output_{j*4+i}.png', obs[i])
    #         cv2.waitKey(100)
    #         # cv2.destroyAllWindows()
    #     obs, reward, done, info  = env.step(2)

    return env

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


class Agent():
    def __init__(self, input_shape):
        self.action_space = env.action_space
        self.replay_memory = deque([], maxlen = 100000)
        self.policy_model = DQNSolver(input_shape = input_shape, n_actions = env.action_space.n).to(device)
        self.target_model = DQNSolver(input_shape = input_shape, n_actions = env.action_space.n).to(device)
        self.target_model.load_state_dict(self.policy_model.state_dict())

        self.training = True
        self.gamma = 0.999
        self.batch_size = 32
        self.TAU = 0.01
        self.step_count = 0
        self.network_sync_rate = 1000
        self.optimzer = optim.Adam(self.policy_model.parameters(), lr = 5e-4)

    def act(self, obs, epsilon):
        if np.random.random() < epsilon and self.training:
            return self.action_space.sample()
        else:
            action_prob = self.policy_model(obs.to(device))
            return torch.argmax(action_prob).item()
    
    def learn(self):
        random_samples = self.sample() # list of Transition
        batch = Transition(*zip(*random_samples)) # Transition of list

        # Normalize states
        state_batch = (torch.tensor(
            np.array(batch.state), dtype=torch.float32
        ) / 255.0).to(device)
        

        # Normalize next_states
        next_state_batch = (torch.tensor(
            np.array(batch.next_state), dtype=torch.float32
        ) / 255.0).to(device)
        
        # state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        # next_state_batch = torch.cat(batch.next_state).to(device)

        state_action_values = self.policy_model(state_batch).gather(1, action_batch.unsqueeze(1))
        # .gather(dim, index) selects elements from the tensor along a given dimension (dim) using the indices provided by index

        with torch.no_grad():
            next_state_value, _ = self.target_model(next_state_batch).max(dim=1)

        expected_state_action_values = reward_batch + self.gamma * next_state_value

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()

        self.step_count += 1
        
        if self.step_count >= self.network_sync_rate:
            self.target_model.load_state_dict(self.policy_model.state_dict())
            self.step_count=0
            # target_net_state_dict = self.target_model.state_dict()
            # policy_net_state_dict = self.policy_model.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
            # self.target_model.load_state_dict(target_net_state_dict)

        return loss.item()

    def sample(self):
        return random.sample(self.replay_memory, self.batch_size)
    
    def save_model(self, episode):
        scripted_model = torch.jit.script(self.policy_model)
        torch.jit.save(scripted_model, f'policy_model_best.pth')
        # torch.jit.save(self.target_model, f'target_model_1.pth')


def train(env, episodes):
    log_data = []
    max_ret = 0
    obs = env.reset()
    agent = Agent(obs.shape)

    for episode in range(1, episodes + 1):
        ret = 0
        done = False
        
        obs_input = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) / 255.0
        # print("Observation shape after transform:", obs.shape)  # (1, 4, 84, 84)

        learn_count = 0
        while not done:
            action = agent.act(obs_input, epsilon=0.05)
            next_obs, reward, done, _ = env.step(action)
            # done = terminanted or truncated
            ret += reward

            action = torch.tensor(action).unsqueeze(0)
            reward = torch.tensor(reward).unsqueeze(0)
            next_obs_input = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0) / 255.0

            agent.replay_memory.append(Transition(obs, action, reward, next_obs))
            obs_input = next_obs_input
            
            if len(agent.replay_memory) < agent.batch_size:
                continue
            # optimize the model
            loss = agent.learn()
            learn_count += 1
            if learn_count % 100 == 0:
                print(f"Episode {episode}, Step {learn_count}, Loss: {loss:.4f}")

        print(f"Episode {episode} Return: {ret}")
        log_data.append({"episode": episode, "return": ret})
        if ret > max_ret:
            agent.save_model(episode)
            max_ret = ret
        # if episode % 10 == 0:
        #     agent.save_model(episode)

        # Save model every 50 episodes
        if episode % 10 == 0:
            with open("training_log.json", "w") as log_file:
                json.dump(log_data, log_file, indent=4)
            print(f"Training log saved at episode {episode}")
        
        obs = env.reset()


if __name__ == '__main__':
    # Initialize environment
    env = make_env()
    # env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='rgb_array')
    # env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # Initialize pygame
    pygame.init()
    # obs = env.reset()

    # print("Available actions and their corresponding indices:")
    for i, action in enumerate(COMPLEX_MOVEMENT):
        print(f"{i}: {action}")
    '''
    0: ['NOOP']
    1: ['right']
    2: ['right', 'A']
    3: ['right', 'B']
    4: ['right', 'A', 'B']
    5: ['A']
    6: ['left']
    7: ['left', 'A']
    8: ['left', 'B']
    9: ['left', 'A', 'B']
    10: ['down']
    11: ['up']
    '''

    episodes = 20000
    train(env, episodes)

    env.close()