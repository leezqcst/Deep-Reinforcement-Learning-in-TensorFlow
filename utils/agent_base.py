# -*- coding:utf-8 -*-
""" base agent class for reinforcement learning

author:zzw922cn
date:2017-4-12
"""
import sys
sys.path.append('../')
sys.dont_write_bytecode = True

from environment_base import Environment

class Agent(object):
  
  def __init__(self, name, lr, epsilon, gamma):
    self.name = name
    self.lr = lr
    self.epsilon = epsilon
    self.gamma = gamma
    self.env = Environment(env_name)

  def pick_action(self):
    pass

  def learn(self):
    pass

