# -*- coding:utf-8 -*-
""" base environment class for reinforcement learning

author:zzw922cn
date:2017-4-12
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import six
import gym
from gym import envs

class Environment(object):
  
  def __init__(self, name):
    self.name = name
    env = gym.make(name)
    self.action_space = env.action_space
    self.action_space_num = env.action_space.n
    self.observation_space = env.observation_space
    self.observation_space_num = env.observation_space.n
    self.env = env

  def reset(self):
    return self.env.reset()

  def render(self):
    return self.env.render()

  def step(self, action):
    if self.action_space.contains(action): 
      observation, reward, done, info = self.env.step(action)
      return observation, reward, done, info
    else:
      raise ValueError('No such action: %s'%str(action))

  @staticmethod
  def get_all_envs():
    return envs.registry.all()
