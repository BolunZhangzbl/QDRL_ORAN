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
from environment import *
from agents_dqn import *
from agents_ddpg import *
from agents_ppo import *

# -- Global Variables


# -- Functions

"""
Main Function to train the RL
"""

def main():
    try:
        env = SlicingEnv()
        agent = BaseAgentDQN(kth=0, state_space=6, action_space=2)
        ep_reward_list = []
        ep_mean_reward_list = []
        avg_reward_list = []
        loss_by_iter_list = []

        for ep in range(num_episodes):

            episodic_reward = 0

            prev_state = env.reset()
            for step in range(max_step):
                tf_prev_state = tf.convert_to_tensor(prev_state)

                action = agent.act(tf_prev_state)

                state, reward, done = env.step(action, 'power')
                episodic_reward += reward
                ep_mean_reward_list.append(reward)

                agent.record((prev_state, action, reward, state))

                # Train the behaviour and target networks
                agent.update()
                loss = agent.update_target()

                loss_by_iter_list.append(loss)
                print(f"inner_iter/episode: {step}/{ep} - "
                      f"Loss: {loss:.6f}")

                if done:
                    print("Done!   inner_iter/episode:{}/{},   Acc. Mean. Reward: {}\n".format(step, ep, np.mean(ep_mean_reward_list)))
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)
            avg_reward = np.mean(ep_mean_reward_list[-last_n:])
            avg_reward_list.append(avg_reward)

            if (np.mean(loss_by_iter_list[-3000:]) <= 1.5e-3):
                break

    finally:
        pass

