# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:38:13 2017

@author: petrichor
"""

import gym
import numpy as np

env=gym.make('FrozenLake-v0')

Q=np.zeros([env.observation_space.n,env.action_space.n])
lr=0.8
y=0.95
num_episodes=2000

rList=[]

for i in range(num_episodes):
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)
print "score over time: "+str(sum(rList)/num_episodes)
print "Final Q-Table Vaalues"
print Q
