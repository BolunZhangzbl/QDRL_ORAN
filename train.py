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

from environment import *
from agents_dqn import *
from agents_ddpg import *
from agents_ppo import *

# -- Global Variables


# -- Functions

"""
Main Function to train the RL
"""
def local_train(agent):
    pass


def global_train():
    env = SlicingEnv()
    agent_power = PowerContrlAgent(Lmin=1, Lmax=4)
    agent_resource = ResourceAllocationAgent(Rmin=0, Rmax=7)
    ep_reward_list = []
    ep_mean_reward_list = []
    avg_reward_list = []
    loss_by_iter_list = []

    try:

        for ep in range(num_episodes):

            episodic_reward = 0

            prev_state = env.reset()
            for step in range(max_step):
                # tf_prev_state = tf.convert_to_tensor(prev_state)

                action_power = np.array([[agent_power.act(prev_state[k][n][:2]) for n in range(4)] for k in range(2)])
                action_resource = np.array([[agent_power.act(prev_state[k][n]) for n in range(4)] for k in range(2)])
                actions = [action_power, action_resource]


                state, rewards, done = env.step(actions)
                episodic_reward += sum(rewards)
                ep_mean_reward_list.append(sum(rewards))

                agent_power.record((prev_state, action_power, rewards[0], state[:, :, :2]))
                agent_resource.record((prev_state, action_resource, rewards[1], state))

                # Train the behaviour and target networks
                agent_power.update()
                loss_power = agent_power.update_target()

                agent_resource.update()
                loss_resource = agent_resource.update_target()

                loss = loss_power + loss_resource

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
        file_path = f"save_dqn/rl/save_lists"
        save_lists(file_path, ep_reward_list, ep_mean_reward_list, avg_reward_list, loss_by_iter_list)

