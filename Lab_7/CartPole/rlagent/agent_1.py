"""agent.py: Contains the entire deep reinforcement learning agent."""
__author__ = "Erik Gärtner"

from collections import deque

import tensorflow as tf
import numpy as np


from .expreplay import ExpReplay


class Agent():
    """
    The agent class where you should implement the vanilla policy gradient agent.
    """

    def __init__(self, tf_session, state_size=(4,), action_size=2,
                 learning_rate=1e-3, gamma=0.99, memory_size=5000):
        """
        The initialization function. Besides saving attributes we also need
        to create the policy network in Tensorflow that later will be used.
        """

        self.state_size = state_size
        self.action_size = action_size
        self.tf_sess = tf_session
        self.gamma = gamma
        self.replay = ExpReplay(memory_size)
        initializer = tf.variance_scaling_initializer()

        with tf.variable_scope('agent'):
            # Create tf placeholders, i.e. inputs into the network graph.
            self.X = tf.placeholder(tf.float32, shape=(None, *state_size))
            self.reward = tf.placeholder(tf.float32, shape=(None,))
            self.action = tf.placeholder(tf.int32, shape=(None,))

            # Create the hidden layers
            hidden = tf.layers.dense(self.X, 5, activation=tf.nn.elu,
                                     kernel_initializer=initializer)
            logits = tf.layers.dense(hidden, action_size, activation=None,
                                     kernel_initializer=initializer)

            self.outputs = tf.nn.softmax(logits)

            # Create the loss. We need to multiply the reward with the
            # log-probability of the selected actions.

            # Create the optimizer to minimize the loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action, logits=logits)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.loss = optimizer.minimize(cross_entropy * self.reward)

        tf_session.run(tf.global_variables_initializer())

    def take_action(self, state):
        """
        Given the current state sample an action from the policy network.
        Return a the index of the action [0..N).
        """
        action = self.tf_sess.run([self.outputs], feed_dict={self.X: [state]})
        return np.random.choice(range(self.action_size), p=action[0][0])

    def record_action(self, state0, action, reward, state1, done):
        """
        Record an action taken by the action and the associated reward
        and next state. This will later be used for traning.
        """
        exp = [state0, action, reward]
        self.replay.add(exp)

    def train_agent(self):
        """
        Train the policy network using the collected experiences during the
        episode(s).
        """
        # Retrieve collected experiences from memory
        exp = self.replay.get_all()

        # Discount and normalize rewards
        states = []
        actions = []
        rewards = []

        for inst in exp:
            states.append(inst[0])
            actions.append(inst[1])
            rewards.append(inst[2])

        rewards = self.discount_rewards_and_normalize(rewards)

        # Shuffle for better learning
        #shuffle = self.replay.sample(self, len(self.replay))

        # Feed the experiences through the network with rewards to compute and
        # minimize the loss.

        self.tf_sess.run([self.loss], feed_dict={self.X: states, self.action: actions, self.reward: rewards})
        self.replay.clear()

    def discount_rewards_and_normalize(self, rewards):
        """
        Given the rewards for an epsiode discount them by gamma.
        Next since we are sending them into the neural network they should
        have a zero mean and unit variance.

        Return the new list of discounted and normalized rewards.
        """
        discount_rate = self.gamma

        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards

        reward_mean = discounted_rewards.mean()
        reward_std = discounted_rewards.std()
        discounted_and_normalized_rewards = (discounted_rewards - reward_mean) / reward_std

        return discounted_and_normalized_rewards