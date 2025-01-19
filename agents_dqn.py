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
from utils import *

# -- Global Variables


# -- Functions

"""
We will design BaseAgent for other algos including DDPG, PPO
"""

# Global Model for FDRL - similar to Critic Network


class GlobalAgentDQN(tf.keras.models.Model):
    def __init__(self, action_space_power, action_space_resource):
        super(GlobalAgentDQN, self).__init__()

        action_space_total = action_space_power + action_space_resource

        # Shared layers for processing both agents' states together
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')

        # Output layer produces a combined action/decision space
        self.output_layer = tf.keras.layers.Dense(action_space_total)

    def call(self, combined_inputs):
        x  = self.d1(combined_inputs)
        x1 = self.d2(x)
        combined_outputs = self.output_layer(x1)

        return combined_outputs


# Base Agent for DQN
class BaseAgentDQN:
    def __init__(self, state_space, action_space, action_mapper,
                 buffer_capacity=int(1e4), batch_size=128):
        self.state_space = state_space
        self.action_space = action_space
        self.action_mapper = action_mapper

        # Buffer
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.state_space))
        self.action_buffer = np.zeros((self.buffer_capacity, 1))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_space))

        # Hyper-parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        # self.loss_func = tf.keras.losses.MeanSquaredError()
        self.loss_func = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.99  # Discount factor

        # Create Deep Q Network
        self.model = self.create_model()
        self.target_model = self.create_model()
        print(self.model.summary())

    def create_model(self):
        input_shape = (self.state_space,)
        X_input = Input(input_shape)
        X = Dense(64, activation="relu")(X_input)
        X = Dense(64, activation="relu")(X)
        X = Dense(self.action_space, activation="linear")(X)
        model = Model(inputs=X_input, outputs=X)
        return model

    def record(self, obs_tuple):
        assert len(obs_tuple) == 4

        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1] - self.action_mapper.minVal
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def act(self, state):
        if state.ndim==1:
            state = np.expand_dims(state, axis=0)

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            action_idx = np.random.choice(self.action_space)
        else:
            q_vals_dist = self.model.predict(state, verbose=0)[0]
            action_idx = tf.argmax(q_vals_dist).numpy()

        action = self.action_mapper.idx_to_action(action_idx)
        return action

    def sample(self):
        sample_indices = np.random.choice(min(self.buffer_counter, self.buffer_capacity), self.batch_size)
        self.sample_indices = sample_indices

        state_sample = tf.convert_to_tensor(self.state_buffer[sample_indices])
        action_sample = tf.convert_to_tensor(self.action_buffer[sample_indices])
        reward_sample = tf.cast(tf.convert_to_tensor(self.reward_buffer[sample_indices]), dtype=tf.float32)
        next_state_sample = tf.convert_to_tensor(self.next_state_buffer[sample_indices])
        return state_sample, action_sample, reward_sample, next_state_sample

    def update(self):
        state_sample, action_sample, reward_sample, next_state_sample = self.sample()
        action_sample_int = tf.cast(tf.squeeze(action_sample), tf.int32)

        # print(f"state_sample: {state_sample}")
        # print(f"action_sample: {action_sample_int}")
        # print(f"reward_sample: {reward_sample}")
        # print(f"next_state_sample: {next_state_sample}")

        target_q_vals = tf.reduce_max(self.target_model(next_state_sample), axis=1)
        y = reward_sample + tf.expand_dims(self.gamma * target_q_vals, axis=1)
        mask = tf.one_hot(action_sample_int, self.action_space)

        with tf.GradientTape() as tape:
            q_vals = self.model(state_sample)
            # print(f"q_vals shape: {q_vals} - {q_vals.shape}")

            # target_q_vals = tf.reduce_max(self.target_model(next_state_sample), axis=1)
            # print(f"target_q_vals shape: {target_q_vals} - {target_q_vals.shape}")

            # y = reward_sample + tf.expand_dims(self.gamma * target_q_vals, axis=1)
            # print(f"reward_sample shape: {reward_sample} - {reward_sample.shape}")
            # print(f"y shape: {y} - {y.shape}")

            # mask = tf.one_hot(action_sample_int, self.action_space)
            # print(f"mask shape: {mask} - {mask.shape}")

            q_action = tf.reduce_sum(tf.multiply(q_vals, mask), axis=1)
            # print(f"q_action shape: {q_action} - {q_action.shape}")

            loss = self.loss_func(y, q_action)
            # print(f"loss shape: {loss} - {loss.shape}\n\n")

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, q_vals

    def update_fedprox(self, global_model):
        global_weight = global_model.model.get_weights()

        state_sample, action_sample, reward_sample, next_state_sample = self.sample()
        action_sample_int = tf.cast(tf.squeeze(action_sample), tf.int32)

        target_q_vals = tf.reduce_max(self.target_model(next_state_sample), axis=1)

        reward_sample = tf.cast(reward_sample, tf.float64)
        y = reward_sample + tf.expand_dims(self.gamma * target_q_vals, axis=1)

        mask = tf.one_hot(action_sample_int, self.action_space)
        mask = tf.cast(mask, tf.float64)

        with tf.GradientTape() as tape:
            q_vals = self.model(state_sample)

            q_action = tf.reduce_sum(tf.multiply(q_vals, mask), axis=1)

            loss = self.loss_func(y, q_action)

            # Prox Term
            prox_term = tf.add_n([tf.reduce_sum(tf.square(w_local-w_global))
                                  for w_local, w_global in zip(self.model.trainable_weights, global_weight)])

            loss += (mu / 2) * tf.cast(prox_term, dtype=loss.dtype)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, q_vals

    # @tf.function
    def update_target(self, tau=0.001):
        # Update target actor
        self.target_model.set_weights(self.model.get_weights())
        # for (a, b) in zip(self.target_model.variables, self.model.variables):
        #     a.assign(b * tau + (1 - tau))

    def save_model_weights(self):
        file_path = f"save_dqn/rl/save_models/model_dqn.keras"
        print(f"filepath: {file_path}")
        self.model.save_model_weights(file_path)

    def load_model_weights(self):
        file_path = f"save_dqn/rl/save_models/model_dqn.keras"
        print(f"filepath: {file_path}")
        self.model.load_model_weights(file_path)


class PowerContrlAgent(BaseAgentDQN):
    def __init__(self, Lmin, Lmax, buffer_capacity=int(1e4), batch_size=32):
        state_space = 3   # [Hn, sum dn, Pk]
        action_space = (Lmax - Lmin + 1)
        action_mapper = ActionMapper(Lmin, Lmax)

        super().__init__(state_space=state_space, action_space=action_space, action_mapper=action_mapper,
                         buffer_capacity=buffer_capacity, batch_size=batch_size)
        self.Lmax = Lmax   # Upper bound for action
        self.Lmin = Lmin   # Lower bound for action


class ResourceAllocationAgent(BaseAgentDQN):
    def __init__(self, Rmin, Rmax, buffer_capacity=int(1e4), batch_size=32):
        state_space = 3   # [Hn, sum dn, num available RBs]
        action_space = (Rmax - Rmin + 1)
        action_mapper = ActionMapper(Rmin, Rmax)

        super().__init__(state_space=state_space, action_space=action_space, action_mapper=action_mapper,
                         buffer_capacity=buffer_capacity, batch_size=batch_size)
        self.Rmax = Rmax   # Upper bound for action
        self.Rmin = Rmin   # Lower bound for action