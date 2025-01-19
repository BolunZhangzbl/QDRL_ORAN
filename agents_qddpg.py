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
            qml.RY(layer_weights[wire, 0]*np.pi, wires=wire)
        for wire in range(0, num_qubits - 1):
            qml.CNOT(wires=[wire, (wire + 1)])

        # for wire in range(0, num_qubits):
        #     qml.CNOT(wires=[wire, (wire+1)%num_qubits])

    @qml.qnode(dev, interface='tf', diff_method='best')
    def qcircuit(inputs, weights):

        qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='Y')

        for layer_weights in weights:
            layer1(layer_weights)

        return [qml.expval(qml.PauliZ(ind)) for ind in range(num_qubits)]

    return qcircuit


# Global Model for FDRL - similar to Critic Network

class QCircuitKeras(tf.keras.models.Model):
    def __init__(self, action_space, **kwargs):
        super(QCircuitKeras, self).__init__(action_space, **kwargs)

        num_qubits = action_space
        dev = qml.device("default.qubit", wires=num_qubits)
        qcircuit = create_circuit(dev, num_qubits)
        weight_shapes = {"weights": (num_layers, num_qubits, 1)}
        self.qmodel = qml.qnn.KerasLayer(qcircuit, weight_shapes, output_dim=num_qubits,
                                         name='qmodel', dtype=tf.float64)
        self.relu = Activation('relu')

    def call(self, inputs):

        outputs = self.qmodel(inputs)
        outputs = self.relu(outputs)
        return outputs


# Base Agent for DQN
class BaseAgentDDPG_Quantum:
    def __init__(self, state_space, action_space, upper_bound, buffer_capacity=int(1e4), batch_size=128):
        self.state_space = state_space
        self.action_space = action_space
        self.upper_bound = upper_bound

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
        self.upper_bound = np.ones((action_space, )) * self.upper_bound

        # Create Actors & Critics

        ### Create Quantum Actor
        self.actor = QCircuitKeras(self.action_space)
        self.target_actor = QCircuitKeras(self.action_space)
        self.target_actor.set_weights(self.actor.get_weights())

        ### Create Classical Critic
        self.critic = self.create_critic()
        self.target_critic = self.create_critic()

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
        action = self.actor.predict(state, verbose=0)

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

        return actor_loss

    @tf.function
    def update_target(self, tau=0.001):
        # Update target actor
        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(b * tau + (1 - tau))

        # Update target critic
        for (a, b) in zip(self.target_critic.variables, self.critic.variables):
            a.assign(b * tau + (1 - tau))


class PowerContrlAgent_Quantum(BaseAgentDDPG_Quantum):
    def __init__(self, Lmin, Lmax, buffer_capacity=int(1e4), batch_size=32):
        state_space = 3  # [Hn, sum dn, Pk]
        action_space = 3
        upper_bound = (Lmax - Lmin + 1)

        super().__init__(state_space=state_space, action_space=action_space, upper_bound=upper_bound,
                         buffer_capacity=buffer_capacity, batch_size=batch_size)
        self.Lmax = Lmax   # Upper bound for action
        self.Lmin = Lmin   # Lower bound for action


class ResourceAllocationAgent_Quantum(BaseAgentDDPG_Quantum):
    def __init__(self, Rmin, Rmax, buffer_capacity=int(1e4), batch_size=32):
        state_space = 3  # [Hn, sum dn]
        action_space = 3
        upper_bound = (Rmax - Rmin + 1)

        super().__init__(state_space=state_space, action_space=action_space, upper_bound=upper_bound,
                         buffer_capacity=buffer_capacity, batch_size=batch_size)
        self.Rmax = Rmax   # Upper bound for action
        self.Rmin = Rmin   # Lower bound for action