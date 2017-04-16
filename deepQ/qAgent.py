# -*- coding:utf-8 -*-
""" base agent class for reinforcement learning

author:zzw922cn
date:2017-4-12
"""
import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import numpy as np
import gym
import matplotlib.pyplot as plt
import time

class QAgent(object):

  def __init__(self, name, lr, epsilon, gamma, env_name):
    self.name = name
    self.lr = lr
    self.epsilon = epsilon
    self.gamma = gamma
    env = gym.make(env_name)

    self.action_space = env.action_space
    self.action_space_num = env.action_space.n
    self.observation_space = env.observation_space
    self.observation_space_num = env.observation_space.n
    self.env = env
    self.qTable = np.zeros([self.observation_space_num, self.action_space_num])

  def reset(self):
    return self.env.reset()

  def render(self, mode='human'):
    self.env.render(mode)

  def close(self):
    self.env.close()

  def pick_action(self, observation, episode, algo='e-epsilon', softmax=True):
    if algo == 'e-epsilon':
      if np.random.uniform() < self.epsilon:
        action_value = self.qTable[observation, :]
        if softmax:
          softmax_fn = lambda x: np.exp(x-np.max(x))/np.exp(x-np.max(x)).sum()
          action_value = map(softmax_fn, action_value)
        np.random.shuffle(action_value)
        action = np.array(action_value).argmax()
      else:
        action = np.random.choice(range(self.action_space_num))
      return action 
    elif algo == 'noisy':
      action_value = self.qTable[observation, :]
      action_value += np.random.randn(self.action_space_num)/(episode+1)
      action = action_value.argmax()
      return action 

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    return observation, reward, done, info

  def learn(self, state, action, reward, new_state):
    time.sleep(0.1)
    update = reward + self.gamma*(self.qTable[new_state,:].max())
        - self.qTable[state, action]
    self.qTable[state, action] += self.lr*update
    
