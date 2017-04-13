# -*- coding:utf-8 -*-
""" base environment class for reinforcement learning

author:zzw922cn
date:2017-4-12
"""
import sys
sys.path.append('../')
sys.dont_write_bytecode = True


import six

@six.add_metaclass(abc.ABCMeta)
class Environment(object):
  
  @property
  def name(self):
    raise NotImplementedError

  @property
  def action_space(self):
    raise NotImplementedError

  @property
  def observation_space(self):
    raise NotImplementedError

  def reset(self):
    raise NotImplementedError

  def render(self):
    raise NotImplementedError

  def step(self, action):
    raise NotImplementedError
