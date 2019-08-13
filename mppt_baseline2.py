import gym
import gym_mppt
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2,DDPG
import argparse
import matplotlib.pyplot as plt







if __name__ == '__main__':
	

	parser = argparse.ArgumentParser('deepid')
	parser.add_argument('--total_timesteps', type=int, default=2000)
	parser.add_argument('--test_steps', type=int, default=2000)
	parser.add_argument('--verbose', type=int, default=0)
	args = parser.parse_args()



	
	#Create the environment:
	env = gym.make('mppt-v0')
	env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
	print('training model')
	# Instantiate the agent:
	model = PPO2(MlpPolicy, env, verbose=args.verbose)
	# Train the agent:
	model.learn(total_timesteps=args.total_timesteps)
	# Save the agent:
	model.save("ppO2_TrainedModel")
	print('Model was succesfull saved')

	obs = env.reset()
	print('state =',obs,obs.shape)

	# Load the trained agent:
	model = PPO2.load('ppO2_TrainedModel')

	#Testing the model:
	for i in range(args.test_steps):
	    action, _states = model.predict(obs)
	    obs, rewards, dones, info = env.step(action)
	    print('state =',obs,'r',rewards,'done', dones, 'info',info)
	    print('vamos bien, por la i=',i)
        if i==(args.test_steps-1):
    	    print('Listo!')
    	    plt.plot(rewards)
    	    plt.show()
            break
       

	    #env.render()