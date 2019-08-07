import gym
import gym_foo
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import argparse


parser = argparse.ArgumentParser('deepid')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--test_epochs', type=int, default=2000)
parser.add_argument('--verbose', type=int, default=0)
args = parser.parse_args()

# env = gym.make('CartPole-v1')
env = gym.make('nessie_end_to_end-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

print('training model')
model = PPO2(MlpPolicy, env, verbose=args.verbose)
model.learn(total_timesteps=args.epochs*700)
obs = env.reset()
for i in range(args.test_epochs*700):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
