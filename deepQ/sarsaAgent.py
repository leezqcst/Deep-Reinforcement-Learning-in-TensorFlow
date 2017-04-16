# -*- coding:utf-8 -*-
""" sarsa agent class for reinforcement learning

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

class SarsaAgent(QAgent):
  def __init__(self, name, lr, epsilon, gamma, env_name):
    QAgent.__init__(self, name, lr, epsilon, gamma, env_name)

  def learn(self, state, action, reward, new_state, new_action):
    update = reward + self.gamma*
        self.qTable[new_state, new_action] - self.qTable[state, action]
    self.qTable[state, action] += self.lr*update
    

