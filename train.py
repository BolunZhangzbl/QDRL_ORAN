# -- Public Imports

# -- Private Imports
import numpy as np

from environment import *
from agents_dqn import *
from agents_qdqn import *

# -- Global Variables
tf.get_logger().setLevel('ERROR')
# tf.keras.backend.set_floatx('float64')

# -- Functions

"""
Main Function to train the RL
"""

def local_train(env, local_models):

    reward_step, loss_step = 0, 0

    for num in range(num_rounds_local):
        # 1. Update traffic across 8 slices
        env.oran.update_ue_traffic()

        # 2. Perform local training across 8 slices
        # i.e., assign rbs to each slice or UEs, given the observation of delay, queue length, and number of available RBs
        for k in range(2):
            for n in range(4):
                local_model = local_models[k*4+n]

                prev_state = np.array(env.reset(k, n))
                action = local_model.act(prev_state)
                state, reward, done = env.step(action, k, n)
                state = np.array(state)
                reward_step += reward/8

                local_model.record((prev_state, action, reward, state))
                loss, q_vals = local_model.update()
                local_model.update_target()
                loss_step += loss/8

                # Assign the updated local model to list
                local_models[k*4+n] = local_model

        # # 3. After allocating RBs, update the queue
        env.oran.update_ue_queue()

        # 4. Clear and release RBs
        env.oran.update_ue_rbs()

    reward_step /= num_rounds_local
    loss_step /= num_rounds_local

    return reward_step, loss_step

# def local_train(env, agent, kth, nth):
#
#     reward_local = 0
#     loss_local = 0
#
#     # env.render()
#     for num in range(num_rounds_local):
#         if num == 0:
#             prev_state = np.array(env.reset(kth, nth))
#         # 1.
#         env.oran.update_ue_traffic()
#
#         action = agent.act(prev_state)
#         state, reward, done = env.step(action, kth, nth)
#         state = np.array(state)
#         reward_local += reward
#
#         agent.record((prev_state, action, reward, state))
#         loss, q_vals = agent.update()
#         agent.update_target()
#         loss_local += loss
#
#         prev_state = state
#
#         # 2.
#         env.oran.update_ue_rbs()
#
#     return reward_local, loss_local, agent


def fed_avg(global_model, local_models):
    avg_weight = []
    local_weights = [local_model.model.get_weights() for local_model in local_models]
    for weight_list_tuple in zip(*local_weights):
        avg_weight.append(np.mean(np.array(weight_list_tuple), axis=0))

    # avg_weight = np.array(avg_weight)

    global_model.model.set_weights(avg_weight)


def train(train_mode='irl', model_type='dnn', save=False):
    assert train_mode in ('irl', 'frl')
    assert model_type in ('qnn', 'dnn')

    env = SlicingEnv()
    agent_class = PowerControlAgent if model_type == 'dnn' else PowerControlAgent_Quantum
    local_models = [agent_class(Lmin=0, Lmax=7) for _ in range(2*4)]

    ep_reward_list = []
    ep_mean_reward_list = []
    avg_reward_list = []
    loss_by_iter_list = []

    if train_mode == 'frl':
        global_model = agent_class(Lmin=0, Lmax=7)

    try:

        for ep in range(num_episodes):

            episodic_reward = 0

            # env.oran.update_ue_rbs()
            for step in range(max_step):

                reward_step, loss_step = local_train(env, local_models)

                # reward_step, loss_step = 0, 0
                # for k in range(2):
                #     for n in range(4):
                #         local_model = local_models[k*4+n]
                #         reward, loss, local_model = local_train(env, local_model, k, n)
                #         local_models[k*4+n] = local_model
                #         reward_step += reward
                #         loss_step += loss
                # reward_step = reward_step / 8
                # loss_step   = loss_step / 8

                episodic_reward += reward_step
                ep_mean_reward_list.append(reward_step)
                loss_by_iter_list.append(loss_step)
                print(f"step/episode: {step}/{ep} - Loss: {loss_step:.6e} - Reward: {reward_step:.6e}")

                if step%25 == 0:
                    # Display RBs info
                    for kth_tmp in range(2):
                        print(f"{'BS':<5}{'rth':<5}{'kth':<5}{'nth':<5}{'mth':<5}{'is_allocated':<15}{'power':<10}")
                        print("-" * 50)
                        for rb in env.oran.BSs[kth_tmp].RBs:
                            print(f"{kth_tmp:<5}{rb.rth:<5}{rb.kth:<5}{rb.nth:<5}{rb.mth:<5}{str(rb.is_allocated):<15}{rb.power:<10.2f}")

                    # Display UEs info
                    print(f"{'service_rate':<15}{'delay':<10}{'packet_size':<15}{'allocated_rbs':<15}")
                    print("-" * 60)
                    UEs = [env.oran.BSs[k].slices[n].UEs[m] for m in range(3) for n in range(4) for k in range(2)]
                    for ue in UEs:
                        print(f"{ue.service_rate:.6f}    {ue.delay:.6f}    {ue.packet_size:<15}{ue.num_rbs_allocated:<10}")

                # 4. We perform global training:
                if train_mode == 'frl':
                    fed_avg(global_model, local_models)
                    global_weight = global_model.model.get_weights()
                    for local_model in local_models:
                        local_model.model.set_weights(global_weight)

            print("Done!   step/episode:{}/{},   Acc. Mean. Reward: {}\n".format(step, ep, np.mean(ep_mean_reward_list)))
            ep_reward_list.append(episodic_reward)
            avg_reward = np.mean(ep_mean_reward_list[-last_n:])
            avg_reward_list.append(avg_reward)

            if (np.mean(loss_by_iter_list[-3000:]) <= 1.5e-3):
                break

    finally:
        if save:
            dir_base = os.path.dirname(os.path.abspath(__file__))
            str_lambda = ''.join(str(val) for val in dict_poisson_lambda.values())
            file_path = os.path.join(dir_base, f"save_dqn/{model_type}/{str_lambda}/{train_mode}/save_lists")
            save_lists(file_path, ep_reward_list, ep_mean_reward_list, avg_reward_list, loss_by_iter_list)


train(train_mode='frl', model_type='dnn', save=True)
# train(train_mode='irl', model_type='qnn', save=True)