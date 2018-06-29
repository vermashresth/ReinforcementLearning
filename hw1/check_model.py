import tensorflow as tf
import gym
import numpy as np

def policy_fn(observation, session, prob = .8):
        graph = tf.get_default_graph()

        input = graph.get_operation_by_name("x").outputs[0]
        prediction = graph.get_tensor_by_name("add:0")
        keep_prob = graph.get_operation_by_name("r").outputs[0]

        return session.run(prediction, feed_dict={input:observation, keep_prob : prob})

def run_exp(envname, render, max_timesteps, num_rollouts):
    with tf.Session() as sess:
        #First let's load meta graph and restore weights
        print "init"
        sess.run(tf.global_variables_initializer())
        print "loading model"
        saver = tf.train.import_meta_graph("my_expert/" + envname + "/" + envname +".meta")
        saver.restore(sess,tf.train.latest_checkpoint("my_expert/" + envname + "/"))
        print "loaded"

        print "loading env"
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit
        print "env loaded"
        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:], sess)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
    print "sessiono over"
    return [returns, observations, actions]



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    outcome = run_exp(args.envname, args.render, args.max_timesteps, args.num_rollouts)
    returns = outcome[0]

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == '__main__':
    main()
