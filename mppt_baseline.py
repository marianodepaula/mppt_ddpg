import gym
import gym_mppt
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import argparse


parser = argparse.ArgumentParser('deepid')
parser.add_argument('--total_timesteps', type=int, default=2000)
parser.add_argument('--test_steps', type=int, default=2000)
parser.add_argument('--verbose', type=int, default=0)
args = parser.parse_args()




env = gym.make('mppt-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

print('training model')
model = PPO2(MlpPolicy, env, verbose=args.verbose)
model.learn(total_timesteps=args.total_timesteps)

print('testing the model')
obs = env.reset()
print('state =',obs,obs.shape)
for i in range(args.test_steps):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print('state =',obs,'r',rewards,'done', dones, 'info',info)
    print('vamos bien, por la i=',i)
    #env.render()