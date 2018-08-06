import load_policy
from check_model import policy_fn as my_policy_fn
import tensorflow as tf
#
# parser = argparse.ArgumentParser()
# parser.add_argument('expert_policy_file', type=str)
#
# print('loading and building expert policy')
# expert_policy_fn = load_policy.load_policy(args.expert_policy_file)
# print('loaded and built')

with tf.Session() as sess:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    args = parser.parse_args()

    #First let's load meta graph and restore weights
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph("my_expert/" + args.envname + "/" + args.envname +".meta")
    saver.restore(sess,tf.train.latest_checkpoint("my_expert/" + args.envname + "/"))

    graph = tf.get_default_graph()
    for op in tf.get_default_graph().get_operations():
        print str(op.name) 
    input_x = graph.get_operation_by_name("x").outputs[0]
    input_y = graph.get_operation_by_name("y").outputs[0]
    prediction = graph.get_tensor_by_name("add:0")
    cost = graph.get_tensor_by_name("mean:0")
