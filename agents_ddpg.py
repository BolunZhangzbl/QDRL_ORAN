# -- Public Imports
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Lambda, Dropout, Concatenate, \
    Reshape, Add, Embedding, Flatten, Conv1D, BatchNormalization, Embedding, LSTM, Activation
from tensorflow.keras.models import Model

# -- Private Imports
from parameters import *

# -- Global Variables


# -- Functions


# Base Agent for DQN
class BaseAgentDDPG:
    def __init__(self, kth, state_space, action_space, upper_bound, buffer_capacity=int(1e4), batch_size=128):
        self.kth = kth  # Index of BS
        self.state_space = state_space
        self.action_space = action_space

        # Buffer
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.state_space))
        self.action_buffer = np.zeros((self.buffer_capacity, self.action_space))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_space))

        # Hyper-parameters
        self.loss_func = tf.keras.losses.Huber()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.99  # Discount factor

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
        assert len(obs_tuple) == 4

        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

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
        next_state_sample = tf.convert_to_tensor(self.next_state_buffer[sample_indices])
        return state_sample, action_sample, reward_sample, next_state_sample

    def update(self):
        state_sample, action_sample, reward_sample, next_state_sample = self.sample()
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_sample, training=True)
            y = reward_sample + self.gamma * self.target_critic([next_state_sample, target_actions], training=True)
            critic_value = self.critic([state_sample, action_sample], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(state_sample, training=True)
            critic_value = self.critic([state_sample, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

    @tf.function
    def update_target(self, tau=0.001):
        # Update target actor
        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(b * tau + (1 - tau))

        # Update target critic
        for (a, b) in zip(self.target_critic.variables, self.critic.variables):
            a.assign(b * tau + (1 - tau))

