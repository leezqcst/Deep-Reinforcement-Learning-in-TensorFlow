# -*- coding:utf-8 -*-
""" sarsa agent runner

author:zzw922cn
date:2017-4-12
"""
import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import numpy as np
import gym
import matplotlib.pyplot as plt

from sarsaAgent import SarsaAgent

if __name__ == '__main__':
  np.random.seed(328922)
  num_episodes = 1500000
  lr = 0.95
  gamma = 0.99
  epsilon = 0.90
  algos = ['e-epsilon', 'noisy']
  env_name = 'FrozenLake-v0'
  sarsaAgent = SarsaAgent(name='sarsa-agent',
             lr = lr,
             epsilon = epsilon,
             gamma=gamma,
             env_name=env_name
           )
  total_r = []
  success_perc = []

  
  for i in range(num_episodes):
    episode_r = 0.
    observation = sarsaAgent.reset()
    action = sarsaAgent.pick_action(observation,episode=i,algo=algos[0])
    count = 0
    while True:
      #sarsaAgent.render()
      new_observation, reward, done, _ = sarsaAgent.step(action)
      new_action = sarsaAgent.pick_action(new_observation,episode=i,algo=algos[0])
      sarsaAgent.learn(observation, action, reward, new_observation, new_action)
      action = new_action
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
  plt.title('sarsa FrozenLake, e-epsilon algorithm')
  plt.show()

  plt.plot(success_perc)
  plt.xlabel('episode')
  plt.ylabel('success percentage')
  plt.show()

  plt.matshow(sarsaAgent.qTable, cmap=plt.cm.gray)
  plt.title('sarsa visualization, state-action pair')
  plt.show()
