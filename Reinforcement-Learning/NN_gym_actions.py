# Aim: Build a simple NN that takes in Obsetvation Array and Outputs two Probabilities i.e. whether to take action for LEFT or RIGHT
# Original Output: Probb of going LEFT
# 1 - Original Probability: Probb. of going RIGHT

import gym
import tensorflow as tf
import numpy as np

# Num Inputs
n_inputs = 4

# Neurons
n_hidden = 4

# Probb to go LEFT
out = 1

intializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden1 = tf.layers.dense(X, n_hidden, activation=tf.nn.relu, kernel_initializer=intializer)
hidden2 = tf.layers.dense(hidden1, n_hidden, activation=tf.nn.relu, kernel_initializer=intializer)

output_layer = tf.layers.dense(hidden2, out, activation=tf.nn.sigmoid, kernel_initializer=intializer)

# Left, Right Probb
# Concatenate tensors along one direction
probb = tf.concat(values=[output_layer, 1-output_layer], axis=1)

# Final Action "0" or "1"
action = tf.multinomial(probb, num_samples=1)

init = tf.global_variables_initializer()

# For 50 episodes of game, take 500 time steps and declare the game as Done.
n_steps = 500
episodes = 50
avg_steps = []
env = gym.make('CartPole-v1')

with tf.Session() as sess:
    sess.run(init)

    for i in range(episodes):
        # Reset Environment
        obs = env.reset()

        for step in range(n_steps):
            # Get the Action: 1D Array
            action_val = action.eval(feed_dict={X: obs.reshape(1,n_inputs)})

            # Feed in the Action: Returns "0" or "1"
            obs, reward, done, info = env.step(action_val [0][0])

            # If the steps are done i.e. done = True
            if done:
                avg_steps.append(step)
                print('Done after {} steps.'.format(step))
                break



print('After {} Episodes, Average Steps per Game was {}'.format(i, np.mean(avg_steps)))
env.close()