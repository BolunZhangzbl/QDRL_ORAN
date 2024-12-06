### Hyper-parameters for O-RAN System Model

bandwidth = 20 * int(1e6)  # 20 MHz
num_rbs = 100
bandwidth_per_rb = bandwidth / num_rbs
num_subcarriers_per_rb = 12
P_max = 30   # 30 dBm for maximum transmission power
tx_rx_ant_gain = 15   # 15 dB for Tx/Rx antenna gain

d_min = 20
d_max = 250

num_gnbs = 2   # Number of BSs
num_ues = 24   # 12 for each service (URLLC, eMBB)
d_inter_gnb = 500

TTI = (1 / 15000) * 2   # Transmission Time Interval (s), subcarrier spacing = 15kHz
N0 = -174   # Noise Power Density (dBm/Hz), Power of AWGN, sigma^2

bits_per_byte = 8

dict_slice_weights = dict(
    urllc2=0.4,
    urllc1=0.3,
    embb2=0.2,
    embb1=0.1,
)

dict_packet_size = dict(
    urllc2=16,
    urllc1=16,
    embb2=32,
    embb1=32,
)

dict_poisson_lambda = dict(
    urllc2=560,
    urllc1=560,
    embb2=1120,
    embb1=1120,
)

dict_reward_done = dict(
    power=10,
    resource=20,
)

alpha = 0.1   # punishment factor for the reward of power control agent


### Hyper-parameters for PPO

max_step = 500
num_episodes = 30
last_n = 50

clip_ratio = 0.2
actor_lr = 3e-4
critic_lr = 1e-3
train_actor_iters = 80
train_critic_iters = 80
lam = 0.97
target_kl = 0.01


"""
Power Control xAPP Agent
"""


"""
Resource Allocation xAPP Agent
"""
