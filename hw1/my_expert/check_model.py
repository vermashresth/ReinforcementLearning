import tensorflow as tf

with tf.Session() as sess:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    args = parser.parse_args()


    #First let's load meta graph and restore weights
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph( args.envname+".ckpt.meta")

    saver.restore(sess,tf.train.latest_checkpoint(''))

    graph = tf.get_default_graph()
    print tf.all_variables()
    input = graph.get_tensor_by_name("Variable:0")
    prediction = graph.get_tensor_by_name("Variable_5/Adam_1:0")
    import numpy as np
    obs = sess.run(prediction, feed_dict={input:np.random.normal(1,0,(11,32))})
    print obs
