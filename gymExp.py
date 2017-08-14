# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:50:16 2017

@author: petrichor
"""

import gym
env=gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())