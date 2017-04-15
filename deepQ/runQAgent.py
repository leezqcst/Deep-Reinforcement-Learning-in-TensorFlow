# -*- coding:utf-8 -*-
""" q agent runner

author:zzw922cn
date:2017-4-12
"""
import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import numpy as np
import gym
import matplotlib.pyplot as plt
from qAgent import QAgent

if __name__ == '__main__':
  np.random.seed(328922)
  num_episodes = 1500000
  lr = 0.95
  gamma = 0.99
  epsilon = 0.90
  algos = ['e-epsilon', 'noisy']
  env_name = 'FrozenLake-v0'
  qAgent = QAgent(name='q-learning-agent',
             lr = lr,
             epsilon = epsilon,
             gamma=gamma,
             env_name=env_name
           )
  total_r = []
  success_perc = []

  
  for i in range(num_episodes):
    episode_r = 0.
    observation = qAgent.reset()
    count = 0
    while True:
      #qAgent.render()
      action = qAgent.pick_action(observation,episode=i,algo=algos[0])
      new_observation, reward, done, _ = qAgent.step(action)
      qAgent.learn(observation, action, reward, new_observation)
      observation = new_observation
      episode_r += reward
      if done:
	break
    total_r.append(episode_r)
    success_perc.append(sum(total_r)/(i+1))
    if i%10000 == 0:
      print('episode:%s, total_reward:%s'%(str(i), str(episode_r)))
      print('successful episode percentage:'+str(sum(total_r)/(i+1)))
  plt.plot(total_r)
  plt.xlabel('episode')
  plt.ylabel('reward')
  plt.title('q learning FrozenLake, e-epsilon algorithm')
  plt.show()

  plt.plot(success_perc)
  plt.xlabel('episode')
  plt.ylabel('success percentage')
  plt.show()

  plt.matshow(qAgent.qTable)
  plt.title('q table visualization, state-action pair', cmap=plt.cm.gray)
  plt.show()
