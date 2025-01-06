# -- Public Imports

# -- Private Imports

from environment import *
from agents_dqn import *
# from agents_ddpg import *
# from agents_ppo import *

# -- Global Variables


# -- Functions

"""
Main Function to train the RL
"""

def local_train(env, agent, kth, nth):

    reward_local = 0
    loss_local = 0

    # env.render()
    for num in range(num_rounds_local):
        if num == 0:
            prev_state = np.array(env.reset(kth, nth))
        # 1.
        env.oran.update_ue_traffic()

        action = agent.act(prev_state)
        state, reward, done = env.step(action, kth, nth)
        state = np.array(state)
        reward_local += reward

        agent.record((prev_state, action, reward, state))
        loss, q_vals = agent.update()
        agent.update_target()
        loss_local += loss

        prev_state = state

        # 2.
        env.oran.update_ue_rbs()

    # print(f"Complete local training for client-({kth},{nth})")

    return reward_local, loss_local


def fed_avg(global_model, local_models):
    avg_weight = []
    local_weights = [local_model.get_weights() for local_model in local_models]
    for weight_list_tuple in zip(*local_weights):
        avg_weight.append(np.mean(np.array(weight_list_tuple), axis=0))

    global_model.set_weights(avg_weight)


def train(train_mode='irl'):
    assert train_mode in ('irl', 'frl', 'crl')

    env = SlicingEnv()
    local_models = [[ResourceAllocationAgent(Rmin=0, Rmax=7) for _ in range(4)] for _ in range(2)]

    ep_reward_list = []
    ep_mean_reward_list = []
    avg_reward_list = []
    loss_by_iter_list = []

    if train_mode == 'frl':
        global_model = ResourceAllocationAgent(Rmin=0, Rmax=7)

    # try:

    for ep in range(num_episodes):

        episodic_reward = 0

        # env.oran.update_ue_rbs()
        for step in range(max_step):

            reward_step, loss_step = 0, 0
            for k in range(2):
                for n in range(4):
                    local_model = local_models[k][n]
                    reward, loss = local_train(env, local_model, k, n)
                    local_models[k][n] = local_model
                    reward_step += reward
                    loss_step += loss
            reward_step = reward_step / 8
            loss_step   = loss_step / 8

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
                global_weight = global_model.get_weights()
                local_models = [local_model.set_weights(global_weight) for local_model in local_models]

        print("Done!   step/episode:{}/{},   Acc. Mean. Reward: {}\n".format(step, ep, np.mean(ep_mean_reward_list)))
        ep_reward_list.append(episodic_reward)
        avg_reward = np.mean(ep_mean_reward_list[-last_n:])
        avg_reward_list.append(avg_reward)

        if (np.mean(loss_by_iter_list[-3000:]) <= 1.5e-3):
            break

    # finally:
    #     file_path = f"save_dqn/rl/save_lists"
    #     save_lists(file_path, ep_reward_list, ep_mean_reward_list, avg_reward_list, loss_by_iter_list)


train(train_mode='irl')