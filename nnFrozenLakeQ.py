# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:49:54 2017

@author: petrichor
"""

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env=gym.make('FrozenLake-v0')

tf.reset_default_graph()

inputs1=tf.placeholder(shape=[1,16],dtype=tf.float32)
W=tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout=tf.matmul(inputs1,W)
predict=tf.argmax(Qout,1)

nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init=tf.initialize_all_variables()

y=0.99
e=0.1
num_episodes=2000

jList=[]
rList=[]

with tf.Session() as sess:
    sess.run(init)
    foor i in range(num_episodes):
        s=env.reset()
        rAll=0
        d=False
        j=0
        while j<99:
            j+=1
            a,allQ=sess.run([predict,Qout],feed_dict={inputs:np.identity(16)[s:s+1]})