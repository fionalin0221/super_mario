from xml.etree import ElementTree as ET
import importlib.util
import sys
import os
import requests
import argparse
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# evaluating
import time
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default="", type=str)
    args = parser.parse_args()
    return args

def run_agent(episode_num=50, time_limit=180, render=False):
    # initializing agent
    agent_path = "student_agent.py"
    module_name = agent_path.replace('.py', '')

    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # makesure the module is set
    spec.loader.exec_module(module)
    Agent = getattr(module, 'Agent')

    os.environ["SDL_AUDIODRIVER"] = "dummy"

    total_reward = 0
    total_time = 0
    agent = Agent()

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    for episode in tqdm(range(episode_num), desc="Evaluating"):
        obs = env.reset()
        start_time = time.time()
        episode_reward = 0
        
        while True:
            action = agent.act(obs)

            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            obs = next_obs

            if time.time() - start_time > time_limit:
                print(f"Time limit reached for episode {episode}")
                break

            if done:
                break

            # Render the environment
            if render:
                env.render()
                time.sleep(0.001)

        end_time = time.time()
        total_reward += episode_reward
        total_time += (end_time - start_time)

    env.close()

    score = total_reward / episode_num
    return score

def eval_score():
    args = parse_arguments()
    
    # retrive submission meta info from the XML file
    xml_file_path = 'meta.xml'

    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Find the 'info' element and extract the 'name' value
    for book in root.findall('info'):
        team_name =  book.find('name').text

    ### Start of evaluation section
    agent_avg_score = run_agent(episode_num=10,
                                time_limit=180,
                                render=False)

    print(f"Final Score: {agent_avg_score}")

    ### End of evaluation section

    # push to leaderboard
    params = {
        'act': 'add',
        'name': team_name,
        'score': str(agent_avg_score),
        'token': args.token
    }
    url = 'http://140.114.89.61/drl_hw3/action.php'

    response = requests.get(url, params=params)
    if response.ok:
        print('Success:', response.text)
    else:
        print('Error:', response.status_code)

if __name__ == '__main__':
    eval_score()