# -*- coding:utf-8 -*-
""" base agent class for reinforcement learning

author:zzw922cn
date:2017-4-12
"""
import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from qAgent import QAgent


class QNetworkAgent(QAgent):
  
  def __init__(self, name, lr, epsilon, gamma, env_name,
        activation_fn=None, 
        loss_fn=tf.squared_difference,
        optimizer_fn=tf.train.GradientDescentOptimizer):

    QAgent.__init__(self, name, lr, epsilon, gamma, env_name)
    self.activation_fn = activation_fn
    self.loss_fn = loss_fn
    self.optimizer = optimizer_fn(self.lr)
    self._build_network()
    self.init_op = tf.global_variables_initializer()


  def pick_action(self, q_predicted, episode, algo='e-epsilon', softmax=True):
    if algo == 'e-epsilon':
      #if (episode+1)%50==0 and self.epsilon<0.99:
        #self.epsilon *= 1.005
      if np.random.uniform() < self.epsilon:
        if softmax:
          softmax_fn = lambda x: np.exp(x-np.max(x))/np.exp(x-np.max(x)).sum()
          q_predicted = map(softmax_fn, q_predicted)
        np.random.shuffle(q_predicted)
        action = np.array(q_predicted).argmax()
      else:
        action = np.random.choice(range(self.action_space_num))
    elif algo == 'noisy':
      q_predicted += np.random.randn(self.action_space_num)/(episode+1)
      action = q_predicted.argmax()
    return action 

  def _build_network(self):
    # define placeholder: state, q_target
    self.state = tf.placeholder(dtype=tf.float32, shape=[1, self.observation_space_num])
    self.q_target = tf.placeholder(dtype=tf.float32, shape=[1, self.action_space_num])

    with tf.variable_scope('q_target_network'):
      self.q_predicted = tf.contrib.layers.fully_connected(
           inputs = self.state,
           num_outputs = self.action_space_num,
           activation_fn = self.activation_fn
      ) 
      
    with tf.variable_scope('loss'):
      self.loss = tf.reduce_mean(self.loss_fn(self.q_target, self.q_predicted))

    with tf.variable_scope('train'):
      self.train_op = self.optimizer.minimize(self.loss)

  def learn(self, q_predicted, action, reward, new_q_predicted, algo='q-learning'): 
    if algo=='q-learning':
      max_new_q = np.max(new_q_predicted)
      q_predicted[0, action] = reward + self.gamma*max_new_q
      self.learned_q_target = q_predicted
