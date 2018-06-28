import load_policy
from check_model import check_model
from bc_cloning import train_model, prepare_data, train_test_split
import tensorflow as tf
import numpy as np



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('envname', type=str)
parser.add_argument('expert_policy_file', type=str)
parser.add_argument('--nb_epochs', type=int, default=10)
parser.add_argument('--nb_iter', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--render', action='store_true')
parser.add_argument('--max_timesteps', type=int)
parser.add_argument('--num_rollouts', type=int, default=20,
                    help='Number of expert roll outs')
args = parser.parse_args()
for iter in range(args.nb_iter):
    # Initialize data
    expert_data = prepare_data(("expert_data_pickles/" + args.envname + "_expert.dict"), False)
    # Train policy on expert demonstrations
    #train_model(init_expert_data, args.envname, args.nb_epochs, args.batch_size)

    # Run trained policy
    check_model(args.envname, args.render, args.max_timesteps, args.num_rollouts)
    outcome = run_exp(args.envname, args.render, args.max_timesteps, args.num_rollouts)
    returns, obs, actions = outcome

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))


    # get expert labels on given obs
    print('loading and building expert policy')
    expert_policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    expert_actions = []
    for o in obs:
        action = expert_policy_fn(o)
        expert_actions.append(action)

    # Data Aggregation step
    expert_data['observations'].extend(obs)
    expert_data['actions'].extend(expert_data)

    split_updated_data = train_test_split(expert_data)

    # Now train our model with this
    train_model(split_updated_data, args.envname, args.nb_epochs, args.batch_size)
    print ("iteration ", iter)










# #First let's load meta graph and restore weights
# sess.run(tf.global_variables_initializer())
# saver = tf.train.import_meta_graph("my_expert/" + args.envname + "/" + args.envname +".meta")
# saver.restore(sess,tf.train.latest_checkpoint("my_expert/" + args.envname + "/"))
#
# graph = tf.get_default_graph()
# for op in tf.get_default_graph().get_operations():
#     print str(op.name)
# input_x = graph.get_operation_by_name("x").outputs[0]
# input_y = graph.get_operation_by_name("y").outputs[0]
# drop_ratio = graph.get_operation_by_name("r").outputs[0]
# prediction = graph.get_tensor_by_name("add:0")
# cost = graph.get_operation_by_name("cost").outputs[0]
# train = tf.get_collection("train")[0]
# _, c=sess.run([train, cost], feed_dict = {input_x:[[10]*11], input_y:[[1]*3],drop_ratio:.8})
# print c
