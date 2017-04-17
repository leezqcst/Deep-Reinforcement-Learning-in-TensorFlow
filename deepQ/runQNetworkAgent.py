# -*- coding:utf-8 -*-
""" QNetworkAgent runner for reinforcement learning

author:zzw922cn
date:2017-4-14
"""
import sys
sys.path.append('../')
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from qNetworkAgent import QNetworkAgent

if __name__ == '__main__':
  np.random.seed(9)
  tf.set_random_seed(22)
  num_episodes = 10000
  lr = 0.15
  gamma = 0.99
  epsilon = 0.9
  algos = ['e-epsilon', 'noisy']
  env_name = 'FrozenLake-v0'
  qNetworkAgent = QNetworkAgent(name='q-network-agent',
             lr = lr,
             epsilon = epsilon,
             gamma=gamma,
             env_name=env_name
           )
  total_r = []
  total_step = []
  total_perc_suc = []
  with tf.Session() as sess:
    sess.run(qNetworkAgent.init_op)
    for episode in range(num_episodes):
      episode_r = 0.
      state = qNetworkAgent.reset()
      count = 0
      while True:
        #qNetworkAgent.render()
        count += 1
        q_predicted = sess.run(qNetworkAgent.q_predicted,
            feed_dict={qNetworkAgent.state: np.eye(1, qNetworkAgent.observation_space_num, state)})

        action = qNetworkAgent.pick_action(q_predicted, episode, algo='e-epsilon')
        new_state, reward, done, _ = qNetworkAgent.step(action)

        new_q_predicted = sess.run(qNetworkAgent.q_predicted,
            feed_dict={qNetworkAgent.state: np.eye(1, qNetworkAgent.observation_space_num, new_state)})

        qNetworkAgent.learn(q_predicted, action, reward, new_q_predicted)

        sess.run(qNetworkAgent.train_op, 
            feed_dict={qNetworkAgent.state: np.eye(1, qNetworkAgent.observation_space_num, state),
                       qNetworkAgent.q_target: qNetworkAgent.learned_q_target}) 

        episode_r += reward
        state = new_state
        if done:
          qNetworkAgent.epsilon = 1-1./((episode/100)+10)
          break
        
      print('episode:'+str(episode)+',reward:'+str(episode_r))
      perc_suc = sum(total_r)/(episode+1)  
      print('percentage of successful episode:'+str(perc_suc))  
      total_r.append(episode_r)
      total_step.append(count)
      total_perc_suc.append(perc_suc)
    print('percentage of successful episode:'+str(sum(total_r)/num_episodes))  
    plt.plot(total_r)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()

    plt.plot(total_perc_suc)
    plt.xlabel('episode')
    plt.ylabel('percentage of success episode')
    plt.show()

    plt.plot(total_step)
    plt.xlabel('episode')
    plt.ylabel('step')
    plt.show()
