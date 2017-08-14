# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:50:16 2017

@author: petrichor
"""

import gym
def moun():
    
    env=gym.make('MountainCar-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())

def cart():
    env=gym.make('CartPole-v0')
    for i_episode in range(20):
        observation=env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
cart()