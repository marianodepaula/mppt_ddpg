import gym
import gym_mppt
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
#import argparse

'''
parser = argparse.ArgumentParser('deepid')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--test_epochs', type=int, default=2000)
parser.add_argument('--verbose', type=int, default=0)
args = parser.parse_args()
'''

epochs = 20
test_epochs=20
verbose = 0 

env = gym.make('mppt-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

print('training model')
model = PPO2(MlpPolicy, env, verbose=verbose)
model.learn(total_timesteps=epochs)

print('testing the model')
obs = env.reset()
for i in range(test_epochs):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print('vamos bien, por la i=',i)
    #env.render()