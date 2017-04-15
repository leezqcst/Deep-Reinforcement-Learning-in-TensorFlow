# -*- coding:utf-8 -*-
""" base agent class for reinforcement learning

author:zzw922cn
date:2017-4-12
"""
import sys
sys.path.append('../')
sys.dont_write_bytecode = True

class Agent(object):
  
  def __init__(self, name, lr):
    self.name = name
    self.lr = lr

  def pick_action(self):
    pass

  def learn(self):
    pass

