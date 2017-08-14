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

r_list=[]

for i in range(num_episodes):
    s=env.reset()
    rAll==0
    d=False
    j=0
    
    while j<99:
        j+=1
        a=np
