# -*- coding:utf-8 -*-
""" base agent class for reinforcement learning

author:zzw922cn
date:2017-4-12
"""
import sys
sys.path.append('../')
sys.dont_write_bytecode = True

from utils.agent_base import Agent
from utils.environment_base import Environment

class QAgent(Agent):

  def __init__(self, name, lr):
    Agent.__init__(self, name, lr)
