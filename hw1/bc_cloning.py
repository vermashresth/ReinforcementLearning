import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import os






def train_test_split(data):
    print data['observations'].shape
    print data['actions'].shape
    data_size = data['observations'].shape[0]
    x, y = data['observations'], data['actions'].reshape(data_size, data['actions'].shape[2])
    print x.shape
    print y.shape
    split_index = int(data_size*.8)
    train_x, train_y = x[:split_index], y[:split_index]
    test_x, test_y = x[split_index:], y[split_index:]
    return train_x, train_y, test_x, test_y


def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob)
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    print out_layer.name
    return out_layer

def prepare_data(expert_file):
    pickle_in = open(expert_file, "rb")
    expert_data = pickle.load(pickle_in)
    return train_test_split(expert_data)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--nb_epochs', type=int, default=8000)
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = prepare_data("expert_data_pickles/" + args.envname + "_expert.dict")

    n_hidden_1 = 32
    n_hidden_2 = 64
    n_hidden_3 = 32
    n_input = train_x.shape[1]
    n_output = train_y.shape[1]

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_output]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }

    keep_prob = tf.placeholder("float", name='r')

    batch_size = 64

    x = tf.placeholder("float", [None, n_input], name = 'x')
    y = tf.placeholder("float", [None, n_output], name = 'y')


    output = multilayer_perceptron(x, weights, biases, keep_prob)
    cost = tf.reduce_mean(tf.square(output - y))
    train = tf.train.AdamOptimizer(0.00006).minimize(cost)
    tf.summary.scalar("cost", cost)


    c_t = []
    c_test = []
    with tf.Session() as sess:
        # Initiate session and initialize all vaiables
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        merge = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter( './logs/' + args.envname + '/train ', sess.graph)
        if(os.path.isdir("my_expert/" + args.envname + "/" )):
            saver.restore(sess,"my_expert/" + args.envname+ "/" + args.envname )
        for epoch in range(args.nb_epochs):
            total_batch = int(len(train_x) / batch_size)
            x_batches = np.array_split(train_x, total_batch)
            y_batches = np.array_split(train_y, total_batch)
            for i in range(total_batch):

                batch_x, batch_y = x_batches[i], y_batches[i]

                summary, _, _ = sess.run([merge, cost, train], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})
                # Run cost and train with each sample
            c_t.append(sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8}))
            train_writer.add_summary(summary, epoch)
            c_test.append(sess.run(cost, feed_dict={x:test_x, y:test_y, keep_prob: 0.8}))
            print('Epoch :',epoch,'training Cost :',c_t[epoch], "testing cost :", c_test[epoch])
        saver.save(sess,"my_expert/" + args.envname+ "/" + args.envname)
        print('Model Saved')


if __name__ == '__main__':
    main()
