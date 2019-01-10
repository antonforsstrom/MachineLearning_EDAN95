"""agent.py: Contains the entire deep reinforcement learning agent."""
__author__ = "Erik Gärtner"

from collections import deque

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
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
        # Added
        self.learning_rate = learning_rate

        with tf.variable_scope('agent'):
            # Create tf placeholders, i.e. inputs into the network graph.
            self.states = tf.placeholder(tf.float32, shape=(None, *state_size))
            self.action = tf.placeholder(tf.int32, shape=(None, ))
            self.reward = tf.placeholder(tf.float32, shape=(None, ))

            # Create the hidden layers
            initializer = variance_scaling_initializer()

            hidden = tf.layers.dense(self.states, self.state_size[0], activation=tf.nn.elu,
                                     kernel_initializer=initializer)
            logits = tf.layers.dense(hidden, self.action_size, activation=None,
                                     kernel_initializer=initializer)
            self.outputs = tf.nn.softmax(logits)

            # self.p_actions = tf.concat(axis=1, values=[self.outputs, 1 - self.outputs])
            # self.actions = tf.multinomial(tf.log(self.p_actions), num_samples=1)

            # Create the loss. We need to multiply the reward with the
            # log-probability of the selected actions.
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action, logits=logits)
            # self.loss = np.sum(np.array(self.outputs * rewards))

            # Create the optimizer to minimize the loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.loss = self.optimizer.minimize(self.cross_entropy * self.reward)

        # Initialize the TensorFlow session
        tf_session.run(tf.global_variables_initializer())

    def take_action(self, state):
        """
        Given the current state, sample an action from the policy network.
        Return a the index of the action [0..N).
        """
        action = self.tf_sess.run([self.outputs], feed_dict={self.states: [state]})
        print(action)
        return np.random.choice(range(self.action_size), p=action[0][0])

    def record_action(self, state0, action, reward, state1, done):
        """
        Record an action taken by the action and the associated reward
        and next state. This will later be used for training.
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
        # implement

        # Feed the experiences through the network with rewards to compute and
        # minimize the loss.
        self.tf_sess.run([self.loss], feed_dict={self.states: [states], self.action: [actions], self.reward: [rewards]})
        self.replay.clear()

    def discount_rewards_and_normalize(self, rewards):
        """
        Given the rewards for an epsiode discount them by gamma.
        Next since we are sending them into the neural network they should
        have a zero mean and unit variance.

        Return the new list of discounted and normalized rewards.
        """
        discounted_rewards = np.zeros(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * self.gamma
            discounted_rewards[step] = cumulative_rewards

        # discounted_and_normalized_rewards = np.zeros(len(discounted_rewards))
        flat_rewards = np.concatenate(discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        discounted_and_normalized_rewards = (discounted_rewards - reward_mean) / reward_std

        return discounted_and_normalized_rewards
