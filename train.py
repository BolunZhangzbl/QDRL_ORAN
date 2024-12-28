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

def local_train(agent):
    pass


def train(train_mode='irl'):
    assert train_mode in ('irl', 'frl', 'crl')

    env = SlicingEnv()
    agent_power = PowerContrlAgent(Lmin=1, Lmax=4)
    agent_resource = ResourceAllocationAgent(Rmin=0, Rmax=7)
    ep_reward_list = []
    ep_mean_reward_list = []
    avg_reward_list = []
    loss_by_iter_list = []

    if train_mode == 'frl':
        agent_global = GlobalAgentDQN(action_space_power=4, action_space_resource=8)

    # try:

    for ep in range(num_episodes):

        episodic_reward = 0

        prev_state = env.reset()
        for step in range(max_step):
            # 0. Render the environments (traffic_update)
            env.render()

            # tf_prev_state = tf.convert_to_tensor(prev_state)

            # 1. We perform the action based on the current state
            prev_state_power = np.sum(prev_state, axis=1)
            action_power = np.array([[agent_power.act(prev_state_power[k])] for k in range(2)])
            action_resource = np.array([[agent_resource.act(prev_state[k][n][:2]) for n in range(4)] for k in range(2)])
            actions_all = (action_power, action_resource)

            # 2. We send the collections of all the actions for both BSs to the environments
            state, rewards, done = env.step(actions_all)
            rewards_power_matrix, rewards_resource_matrix = rewards[0], rewards[1]
            rewards_total = np.sum(rewards_power_matrix) + np.sum(rewards_resource_matrix)
            episodic_reward += rewards_total
            ep_mean_reward_list.append(rewards_total)

            # 3. We perform local training:
            loss = 0
            for k in range(2):
                agent_power.record((prev_state_power[k], action_power[k], rewards_power_matrix[k][0], np.sum(state[k], axis=0)))

                # Train the behaviour and target networks
                loss_power, q_vals_power = agent_power.update()
                agent_power.update_target()

                loss += loss_power
                # print(f"loss_power: {loss_power}")

                for n in range(4):
                    agent_resource.record((prev_state[k][n][:2], action_resource[k][n], rewards_resource_matrix[k][n], state[k][n][:2]))

                    loss_resource, q_vals_resource = agent_resource.update()
                    agent_resource.update_target()

                    loss += loss_resource
                    # print(f"loss_resource: {loss_resource}")

            loss_by_iter_list.append(loss)
            print(f"inner_iter/episode: {step}/{ep} - Loss: {loss:.6e}")
            if step%100 == 0:
                print(f"{'rth':<5}{'kth':<5}{'nth':<5}{'mth':<5}{'is_allocated':<15}{'power':<10}")
                print("-" * 45)
                for rb in env.oran.RBs:
                    print(f"{rb.rth:<5}{rb.kth:<5}{rb.nth:<5}{rb.mth:<5}{str(rb.is_allocated):<15}{rb.power:<10.2f}")

                print(
                    f"{'traffic_curr':<15}{'delay':<10}{'packet_size':<15}{'allocated_rbs':<15}")
                print("-" * 75)
                UEs = [env.oran.BSs[k].slices[n].UEs[m] for m in range(3) for n in range(4) for k in range(2)]
                for ue in UEs:
                    print(
                        f"{ue.traffic_curr:<15}{ue.delay:<10}{ue.packet_size:<15}{ue.num_rbs_allocated:<15}")

            # 4. We perform global training:

            if done:
                print("Done!   inner_iter/episode:{}/{},   Acc. Mean. Reward: {}\n".format(step, ep, np.mean(ep_mean_reward_list)))
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)
        avg_reward = np.mean(ep_mean_reward_list[-last_n:])
        avg_reward_list.append(avg_reward)

        if (np.mean(loss_by_iter_list[-3000:]) <= 1.5e-3):
            break

    # finally:
    #     file_path = f"save_dqn/rl/save_lists"
    #     save_lists(file_path, ep_reward_list, ep_mean_reward_list, avg_reward_list, loss_by_iter_list)


train(train_mode='irl')