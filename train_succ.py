import cv2
import pygame
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T

import gym
import gym_super_mario_bros
from gym.spaces import Box
from gym.wrappers.frame_stack import LazyFrames
from gym.wrappers import FrameStack, ResizeObservation, GrayScaleObservation
# from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from nes_py.wrappers import JoypadSpace
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
def convert_obs(obs):
    """Ensure the observation is always a NumPy array."""
    if isinstance(obs, LazyFrames):
        obs = np.array(obs)  # Convert LazyFrames to a NumPy array
    elif isinstance(obs, torch.Tensor):
        obs = obs.numpy()    # Convert PyTorch tensor to NumPy array
    return obs

# # permute, to tensor(float) and turn to gray scale
# class GrayScaleObservation(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         obs_shape = self.observation_space.shape[:2]  #(240, 256)
#         self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

#     def permute_orientation(self, observation):
#         # permute [H, W, C] array to [C, H, W] tensor
#         observation = np.transpose(observation, (2, 0, 1))
#         observation = torch.tensor(observation.copy(), dtype=torch.float)
#         return observation

#     def observation(self, observation):
#         observation = self.permute_orientation(observation)
#         transform = T.Grayscale()
#         observation = transform(observation)
#         return observation

# # resize to 84*84, normalize to [0, 1]
# class ResizeObservation(gym.ObservationWrapper):
#     def __init__(self, env, shape):
#         super().__init__(env)
#         if isinstance(shape, int):
#             self.shape = (shape, shape)
#         else:
#             self.shape = tuple(shape)

#         obs_shape = self.shape + self.observation_space.shape[2:]
#         self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

#     def observation(self, observation):
#         transforms = T.Compose(
#             [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
#         )
#         observation = transforms(torch.tensor(observation)).squeeze(0)
#         return observation

# class ProcessImage(gym.ObservationWrapper):
#     def __init__(self, env, width = 84, height = 84):
#         super(ProcessImage, self).__init__(env)
#         self._width = width
#         self._height = height

#         original_space = self.observation_space
#         self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, self._height, self._width), dtype=np.float32)
    
#     def observation(self, obs):
#         # print(obs.shape, type(obs))
#         frame = ((obs[0]+obs[1]+obs[2]) / 3).astype(np.uint8)
#         # frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
#         frame = cv2.resize(frame, (84, 90), interpolation=cv2.INTER_AREA)
#         crop_h = (90 - self._height) // 2
#         frame = frame[crop_h:crop_h + self._height, :]
#         frame = np.expand_dims(frame, axis=0)
#         return frame

def make_env():
    # env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='rgb_array')
    env = gym_super_mario_bros.make('SuperMarioBros-v0') # (240, 256, 3) class <'numpy.ndarray'>
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)

    # env = GrayScaleObservation(env) # Convert to grayscale and remove the color channel
    # env = ResizeObservation(env, shape=84) # Resize to 84×84 #<class 'torch.Tensor'>
    # env = FrameStack(env, num_stack=4)  # Stack last 4 frames
    
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayScaleObservation(env, keep_dim=False) #<class 'numpy.ndarray'>
    env = FrameStack(env, num_stack=4)
    
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
        # self.replay_memory = deque([], maxlen = 50000)
        self.replay_memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.policy_model = DQNSolver(input_shape = input_shape, n_actions = env.action_space.n).to(device)
        self.target_model = DQNSolver(input_shape = input_shape, n_actions = env.action_space.n).to(device)
        # self.policy_model = torch.jit.load("policy_model_latest.pth")
        # self.target_model = torch.jit.load("policy_model_latest.pth")
        for param in self.target_model.parameters():
            param.requires_grad = False
        # self.target_model.load_state_dict(self.policy_model.state_dict())

        # training
        self.training = True
        self.gamma = 0.9
        self.batch_size = 32
        # self.TAU = 0.01
        self.step_count = 0
        self.network_sync_rate = 10000
        self.optimzer = optim.Adam(self.policy_model.parameters(), lr = 0.00025) #lr=0.00025
        self.criterion = nn.SmoothL1Loss()

        # exploration
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.exploration_fraction = 0.2
        self.total_timestep = 10000000

        # information buffer
        self.losses = []
        self.rewards = []
        self.qvalues = []
        self.reward_mean = []
        self.qvalues_mean = []
        self.returns = []

        # Debug
        # self.prev_batch = None

    def linear_schedule(self, start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    def act(self, state, epsilon):
        state = state.to(device)
        qvalues = self.policy_model(state)
        action = torch.argmax(qvalues).item()

        # only for plot
        self.qvalues.append(torch.max(qvalues).item())

        # epsilon-greedy
        if np.random.rand() < epsilon and self.training:
            return self.action_space.sample()
        else:
            return action

    def cache(self, state, next_state, action, reward, done):
        # def first_if_tuple(x):
        #     return x[0] if isinstance(x, tuple) else x
        # state = first_if_tuple(state).__array__()
        # next_state = first_if_tuple(next_state).__array__()

        state = state.__array__()
        next_state = next_state.__array__()

        # state = torch.tensor(state, dtype=torch.float32)
        # next_state = torch.tensor(next_state, dtype=torch.float32)

        state = torch.tensor(state, dtype=torch.float32) / 255
        next_state = torch.tensor(next_state, dtype=torch.float32) / 255
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.replay_memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    
    def recall(self):
        batch = self.replay_memory.sample(self.batch_size).to(device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        # random_samples = self.sample() # list of Transition
        # batch = Transition(*zip(*random_samples)) # Transition of list

        # # Normalize states
        # state_batch = (torch.tensor(np.array(batch.state), dtype=torch.float) / 255.0).to(device)
        
        # # Normalize next_states
        # next_state_batch = (torch.tensor(np.array(batch.next_state), dtype=torch.float) / 255.0).to(device)
        
        # action_batch = torch.cat(batch.action).to(device)
        # reward_batch = torch.cat(batch.reward).to(device)
        # done_batch = torch.cat(batch.done).to(device)
        
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = self.recall()

        # # Normalize states
        # state_batch = (torch.tensor(np.array(batch.state), dtype=torch.float) / 255.0).to(device)
        
        # # Normalize next_states
        # next_state_batch = (torch.tensor(np.array(batch.next_state), dtype=torch.float) / 255.0).to(device)
        

        # TD estimate
        state_action_values = self.policy_model(state_batch).gather(1, action_batch.unsqueeze(1))
        # .gather(dim, index) selects elements from the tensor along a given dimension (dim) using the indices provided by index

        # TD target
        with torch.no_grad():
            next_state_value, _ = self.target_model(next_state_batch).max(dim=1)

        expected_state_action_values = reward_batch + (1-done_batch.float()) * self.gamma * next_state_value

        # loss function
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # back-propagation
        self.optimzer.zero_grad()
        loss.backward()

        # clamp gradient
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        # update weight
        self.optimzer.step()

        self.step_count += 1
        
        if self.step_count >= self.network_sync_rate:
            self.target_model.load_state_dict(self.policy_model.state_dict())
            self.step_count=0

        return loss.item()

    # def sample(self):
    #     return random.sample(self.replay_memory, self.batch_size)

    def clearbuffer(self):
        self.losses = []
        self.rewards = []
        self.qvalues = []
    
    def save_model(self, episode=0, file_name="policy_model_best.pth"):
        scripted_model = torch.jit.script(self.policy_model)
        torch.jit.save(scripted_model, file_name)
        # torch.jit.save(self.target_model, f'target_model_1.pth')


def train(env, episodes):
    log_data = []
    max_ret = 0
    obs = env.reset()

    agent = Agent(obs.shape)

    # The global timestep and progress_bar is for epsilon scheduling and progress visualization
    global_timestep = 0
    progress_bar = tqdm(total=agent.total_timestep, desc="Training Progress")

    for episode in range(1, episodes + 1):
        ret = 0
        done = False
        
        obs_input = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0) / 255
        # obs_input = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) / 255.0
        
        # print("Observation shape after transform:", obs.shape)  # (1, 4, 84, 84)

        learn_count = 0
        while True:
            # get epsilon from epsilon-scheduler, depends on the curent global-timestep
            epsilon = agent.linear_schedule(agent.epsilon_start, agent.epsilon_end, agent.exploration_fraction * agent.total_timestep, global_timestep)
            
            # choose action by epsilon-greedy
            action = agent.act(obs_input, epsilon)
            
            # apply action to environment and get r and s'
            next_obs, reward, done, info = env.step(action)
            

            ret += reward
            agent.cache(obs, next_obs, action, reward, done)

            # action = torch.tensor(action).unsqueeze(0)
            # reward = torch.tensor(reward).unsqueeze(0)
            # done = torch.tensor(done).unsqueeze(0)

            next_obs_input = torch.tensor(np.array(next_obs), dtype=torch.float32).unsqueeze(0) / 255
            # next_obs_input = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0) / 255.0

            # agent.replay_memory.append(Transition(obs, action, reward, next_obs, done))
            # agent.append(Transition(obs, action, reward, next_obs))
            
            obs_input = next_obs_input
            obs = next_obs
            
            # env.render()
            
            if len(agent.replay_memory) < agent.batch_size:
                continue

            # optimize the model
            loss = agent.learn()
            
            learn_count += 1
            global_timestep += 1

            # for plot
            agent.losses.append(loss)
            agent.rewards.append(reward)

            # log info
            if learn_count % 1000 == 0:
                tqdm.write(f"Episode {episode}, Step {learn_count}, Loss: {loss:.4f}, Epsilon: {epsilon}")

            # Update tqdm bar manually
            progress_bar.update(1)

            # Check if end of game
            if done or info["flag_get"]:
                break


        agent.reward_mean.append(np.mean(agent.rewards))
        agent.qvalues_mean.append(np.mean(agent.qvalues))
        agent.returns.append(ret)

        tqdm.write(f"Episode {episode} Return: {ret}, Epsilon: {epsilon}")
        log_data.append({"episode": episode, "return": ret})
        
        if ret > max_ret:
            agent.save_model(episode, file_name='policy_model_best.pth')
            max_ret = ret

        # Save model every 50 episodes
        if episode % 20 == 0:
            agent.save_model(episode, file_name='policy_model_latest.pth')
            tqdm.write("[INFO]: Save model!")

            with open("training_log.json", "w") as log_file:
                json.dump(log_data, log_file, indent=4)
            tqdm.write(f"Training log saved at episode {episode}")
            save_plot(episode, agent.losses, agent.reward_mean, agent.qvalues_mean, agent.returns)
        
        agent.clearbuffer()
        obs = env.reset()

    progress_bar.close()

def moving_average(values, n):
    offset = (n - 1) // 2
    v = [values[0]] * offset + values + [values[-1]] * offset
    return [sum(v[i - offset : i + offset + 1]) / n for i in range(offset, len(v) - offset)]

def save_plot(episode, losses, rewards, qvalues, returns):
    fig, axis = plt.subplots(2, 3, figsize=(16, 5))
    axis = axis.flatten()

    axis[0].plot(range(len(losses)), losses)
    axis[0].set_ylabel('Loss per optimization')
    axis[1].plot(range(len(rewards)), rewards)
    axis[1].set_ylabel('Average of the reward per episode')
    axis[2].plot(range(len(qvalues)), qvalues)
    axis[2].set_ylabel('Average of the max predicted Q value')
    axis[3].plot(range(len(returns)), returns)
    axis[3].set_ylabel('Return per episode')

    returns_movavg = moving_average(returns, 60)
    axis[3].plot(range(len(returns_movavg)), returns_movavg, color='red')

    fig.suptitle(f"Episode {episode}")
    fig.tight_layout()
    plt.savefig(f"plot/training3/episode-{episode}.png")
    tqdm.write(f"Figure \"episode-{episode}.png\" saved.")
    for axis in axis:
        axis.cla()

if __name__ == '__main__':
    # Initialize environment
    env = make_env()
    # Initialize pygame
    pygame.init()

    # print("Available actions and their corresponding indices:")
    # for i, action in enumerate(COMPLEX_MOVEMENT):
    #     print(f"{i}: {action}")
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