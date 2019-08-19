import gym
import gym_mppt
import numpy as np
#from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines import PPO2,DDPG
import argparse
import matplotlib.pyplot as plt
import pickle



class DATOS(object):


	def __init__(self,init_state,Temp_0,Irr_0,accion_0=0.):

		v=init_state[0]
		self.V = list([v])
		p=init_state[1]
		self.P = list([p])
		deltav=init_state[2]
		self.deltaV = list([deltav])
		self.I = list([0.])
		self.Temp = list([Temp_0])
		self.Irr = list([Irr_0])
		self.acciones = list([accion_0])
		
	def add(self,v,p,dv,i,T,irr,accion):
		self.V.append(v)
		self.P.append(p)
		self.deltaV.append(dv)
		self.I.append(i)
		self.Temp.append(T)
		self.Irr.append(irr)
		self.acciones.append(accion)



	def plotear(self):
		plt.plot(self.V,self.P)
		plt.xlabel('V (v)')
		plt.ylabel('P (w)')
		plt.title('V-P curve')
		plt.savefig('VPcurve' + '.png')
		plt.show()

		plt.plot(self.V,self.I)
		plt.xlabel('V (v)')
		plt.ylabel('I (A)')
		plt.title('V-I curve')
		plt.savefig('VIcurve' + '.png')
		plt.show()


		plt.plot(self.V)
		plt.xlabel('t')
		plt.ylabel('V (v)')
		plt.savefig('Tesion' + '.png')
		plt.show()

		plt.plot(self.I)
		plt.xlabel('t')
		plt.ylabel('I (a)')
		plt.savefig('Corriente' + '.png')
		plt.show()

		plt.plot(self.P)
		plt.xlabel('t')
		plt.ylabel('P (w)')
		plt.savefig('Potencia' + '.png')
		plt.show()

		plt.plot(self.acciones)
		plt.xlabel('t')
		plt.ylabel('acciones (\deltaV)')
		plt.title('actions')
		plt.savefig('Acciones' + '.png')
		plt.show()

		plt.plot(self.Temp)
		plt.xlabel('t')
		plt.ylabel('(ºC)')
		plt.title('Temperature profile')
		plt.savefig('Temperatura' + '.png')
		plt.show()

		plt.plot(self.Irr)
		plt.xlabel('t')
		plt.ylabel('(Irradiance)')
		plt.title('Solar irradiance profile')
		plt.savefig('Irradiancia' + '.png')
		plt.show()