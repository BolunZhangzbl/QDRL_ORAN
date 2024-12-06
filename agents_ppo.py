# -- Public Imports
import os
import random
from scipy import signal
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Lambda, Dropout, Concatenate, \
    Reshape, Add, Embedding, Flatten, Conv1D, BatchNormalization, Embedding, LSTM, Activation
from tensorflow.keras.models import Model

# -- Private Imports
from parameters import *

# -- Global Variables


# -- Functions

def discounted_cumulative_sums(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.keras.ops.log_softmax(logits)
    logprobability = tf.keras.ops.sum(
        tf.keras.ops.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Base Agent for DQN
class BaseAgentPPO:
    def __init__(self, kth, state_space, action_space, upper_bound, buffer_capacity=int(1e4), batch_size=128):
        self.kth = kth  # Index of BS
        self.state_space = state_space
        self.action_space = action_space

        # Buffer
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.trajectory_start_index = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.state_space))
        self.action_buffer = np.zeros((self.buffer_capacity, self.action_space))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.advantage_buffer = np.zeros((self.buffer_capacity, 1))
        self.return_buffer = np.zeros((self.buffer_capacity, 1))
        self.value_buffer = np.zeros((self.buffer_capacity, 1))
        self.logprob_buffer = np.zeros((self.buffer_capacity, 1))

        # Hyper-parameters
        self.loss_func = tf.keras.losses.Huber()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.99  # Discount factor
        self.lam = 0.95

        self.lower_bound = np.zeros((state_space, ))
        self.upper_bound = np.ones((action_space, )) * upper_bound

        # Create Actors & Critics
        self.actor = self.create_actor()
        self.target_actor = self.create_actor()
        print(self.actor.summary())

        self.critic = self.create_critic()
        self.target_critic = self.create_critic()
        print(self.critic.summary())

    def create_actor(self):
        input_shape = (self.state_space,)
        X_input = Input(input_shape)
        X = Dense(64, activation="relu")(X_input)
        X = Dense(64, activation="relu")(X)
        X = Dense(self.action_space, activation="linear")(X)
        model = Model(inputs=X_input, outputs=X)
        return model

    def create_critic(self):
        # State
        state_input = Input((self.state_space, ))
        s1 = Dense(64, activation='relu')(state_input)

        # Action
        action_input = Input((self.action_space, ))
        a1 = Dense(64, activation='relu')(action_input)

        # Concat
        concat = Concatenate()([s1, a1])

        c1 = Dense(256, activation='relu')(concat)
        c2 = Dense(256, activation='relu')(c1)
        outputs = Dense(1, activation='linear')(c2)

        model = Model(inputs=[state_input, action_input], outputs=outputs)

        return model

    def record(self, obs_tuple):
        assert len(obs_tuple) == 5

        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.value_buffer[index] = obs_tuple[3]
        self.logprob_buffer[index] = obs_tuple[4]
        self.buffer_counter += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing the advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.buffer_capacity)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma*values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(deltas, self.gamma*self.lam)
        self.return_buffer[path_slice] = discounted_cumulative_sums(rewards, self.gamma)[:-1]

        self.trajectory_start_index = self.buffer_counter

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.buffer_counter, self.trajectory_start_index = 0, 0
        advantage_mean, advatange_std = (np.mean(self.advantage_buffer), np.std(self.advantage_buffer))
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advatange_std

        return (self.state_buffer, self.action_buffer, self.advantage_buffer,
                self.return_buffer, self.logprob_buffer)

    def act(self, state):
        # Deterministic Policy
        action = self.actor.predict(state)

        # Clip
        action = np.clip(action, self.lower_bound, self.upper_bound)

        return action

    def sample(self):
        sample_indices = np.random.choice(min(self.buffer_counter, self.buffer_capacity), self.batch_size)
        state_sample = tf.convert_to_tensor(self.state_buffer[sample_indices])
        action_sample = tf.convert_to_tensor(self.action_buffer[sample_indices])
        reward_sample = tf.cast(tf.convert_to_tensor(self.reward_buffer[sample_indices]), dtype=tf.float32)

        return state_sample, action_sample, reward_sample

    @tf.function
    def update_actor(self):
        state_buffer, action_buffer, advantage_buffer, return_buffer, logprob_buffer = self.get()

        with tf.GradientTape() as tape:
            actions = self.actor(state_buffer)
            log_probs = logprobabilities(actions, action_buffer)
            ratio = tf.keras.ops.exp(log_probs - logprob_buffer)
            min_advantage = tf.keras.ops.where(advantage_buffer > 0,
                                            (1 + clip_ratio) * advantage_buffer,
                                            (1 - clip_ratio) * advantage_buffer,)
            policy_loss = -tf.keras.ops.mean(tf.keras.ops.minimum(ratio * advantage_buffer, min_advantage))

        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.keras.ops.mean(logprob_buffer - logprobabilities(self.actor(state_buffer), action_buffer))
        kl = tf.keras.ops.sum(kl)

        return kl

    @tf.function
    def update_critic(self):
        state_buffer, action_buffer, advantage_buffer, return_buffer, logprob_buffer = self.get()

        with tf.GradientTape() as tape:
            value_loss = tf.keras.ops.mean((return_buffer - self.critic(state_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    @tf.function
    def update_target(self, tau=0.001):
        # Update target actor
        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(b * tau + (1 - tau))

        # Update target critic
        for (a, b) in zip(self.target_critic.variables, self.critic.variables):
            a.assign(b * tau + (1 - tau))

