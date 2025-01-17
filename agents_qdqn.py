# -- Public Imports
import os
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Lambda, Dropout, Concatenate, \
    Reshape, Add, Embedding, Flatten, Conv1D, BatchNormalization, Embedding, LSTM, Activation
from tensorflow.keras.models import Model

import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers, BasicEntanglerLayers

# -- Private Imports
from parameters import *
from utils import *

# -- Global Variables


# -- Functions

"""
We will design BaseAgent for other algos including DDPG, PPO
"""

def create_circuit(dev, num_qubits):
    if dev is None:
        dev = qml.device("default.qubit", wires=num_qubits)

    def layer1(layer_weights):
        """
        q layer for fading:
        :param layer_weights:
        :return:
        """
        for wire in range(num_qubits):
            qml.RY(layer_weights[wire, 0] * np.pi, wires=wire)
        for wire in range(0, num_qubits - 1):
            qml.CNOT(wires=[wire, (wire + 1)])

    @qml.qnode(dev, interface='tf', diff_method='best')
    def qcircuit(inputs, weights):

        qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='X')

        for layer_weights in weights:
            layer1(layer_weights)

        return qml.probs(wires=range(num_qubits))

    return qcircuit


# Global Model for FDRL - similar to Critic Network

class QCircuitKeras(tf.keras.models.Model):
    def __init__(self, action_space, **kwargs):
        super(QCircuitKeras, self).__init__(action_space, **kwargs)

        num_qubits = int(math.log2(action_space))
        dev = qml.device("default.qubit", wires=num_qubits)
        qcircuit = create_circuit(dev, num_qubits)
        weight_shapes = {"weights": (num_layers, num_qubits, 1)}
        self.qmodel = qml.qnn.KerasLayer(qcircuit, weight_shapes, output_dim=num_qubits,
                                         name='qmodel', dtype=tf.float64)
        # self.d1 = Dense(3, activation='relu')
        # self.d2 = Dense(8, activation='linear')

    def call(self, inputs):

        # inputs = self.d1(inputs)
        outputs = self.qmodel(inputs)
        # outputs = self.d2(outputs)
        return outputs


# Base Agent for DQN
class BaseAgentDQN_Quantum:
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

        # Create Quantum Deep Q Network
        self.model = QCircuitKeras(self.action_space)
        self.target_model = QCircuitKeras(self.action_space)

    def record(self, obs_tuple):
        assert len(obs_tuple) == 4

        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1] - self.action_mapper.minVal   # Only record the indices of action
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

        reward_sample = tf.cast(reward_sample, tf.float64)
        y = reward_sample + tf.expand_dims(self.gamma * target_q_vals, axis=1)

        mask = tf.one_hot(action_sample_int, self.action_space)
        mask = tf.cast(mask, tf.float64)

        with tf.GradientTape() as tape:
            q_vals = self.model(state_sample)

            q_action = tf.reduce_sum(tf.multiply(q_vals, mask), axis=1)

            loss = self.loss_func(y, q_action)

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


class PowerContrlAgent_Quantum(BaseAgentDQN_Quantum):
    def __init__(self, Lmin, Lmax, buffer_capacity=int(1e4), batch_size=32):
        state_space = 3   # [Hn, sum dn, Pk]
        action_space = (Lmax - Lmin + 1)
        action_mapper = ActionMapper(Lmin, Lmax)

        super().__init__(state_space=state_space, action_space=action_space, action_mapper=action_mapper,
                         buffer_capacity=buffer_capacity, batch_size=batch_size)
        self.Lmax = Lmax   # Upper bound for action
        self.Lmin = Lmin   # Lower bound for action


class ResourceAllocationAgent_Quantum(BaseAgentDQN_Quantum):
    def __init__(self, Rmin, Rmax, buffer_capacity=int(1e4), batch_size=32):
        state_space = 3   # [Hn, sum dn]
        action_space = (Rmax - Rmin + 1)
        action_mapper = ActionMapper(Rmin, Rmax)

        super().__init__(state_space=state_space, action_space=action_space, action_mapper=action_mapper,
                         buffer_capacity=buffer_capacity, batch_size=batch_size)
        self.Rmax = Rmax   # Upper bound for action
        self.Rmin = Rmin   # Lower bound for action