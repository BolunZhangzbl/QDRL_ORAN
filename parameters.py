### Hyper-parameters for O-RAN System Model

bandwidth = 20 * int(1e6)  # 20 MHz
num_rbs = 100
bandwidth_per_rb = bandwidth / num_rbs
# bandwidth_per_rb = int(1.8e4)   # 180 kHz per RB
num_subcarriers_per_rb = 12
P_max = 40   # 40 dBm for maximum transmission power
tx_rx_ant_gain = 15   # 15 dB for Tx/Rx antenna gain

d_min = 20    # Minimum distance between BS and UE
d_max = 500   # Maximum distance between BS and UE

num_gnbs = 2   # Number of BSs
num_ues = 24   # 12 for each service (URLLC, eMBB)
d_inter_gnb = 500   # Distance between 2 BSs

TTI = (1 / 15000) * 2   # Transmission Time Interval (s), subcarrier spacing = 15kHz

N0 = 3.98*1e-15   # Noise Power Density (W/Hz), Power of AWGN, sigma^2
# N0 = -114   # Noise Power Density (dBm)

bits_per_byte = 8

dict_slice_weights = dict(
    urllc2=0.25,
    urllc1=0.25,
    embb2=0.25,
    embb1=0.25,
)

dict_packet_size = dict(
    urllc2=16,
    urllc1=16,
    embb2=32,
    embb1=32,
)

dict_poisson_lambda = dict(
    urllc2=40,
    urllc1=40,
    embb2=80,
    embb1=80
)

dict_reward_done = dict(
    power=10,
    resource=20,
)

alpha = 0.1   # punishment factor for the reward of power control agent


### Hyper-parameters for MADQN

max_step = 50
num_episodes = 30
num_rounds_local = 1

last_n = 10

clip_ratio = 0.2
actor_lr = 3e-4
critic_lr = 1e-3
train_actor_iters = 80
train_critic_iters = 80
lam = 0.97
target_kl = 0.01


### Hyper-parameters for Quantum DQN
num_layers = 3
num_qubits = 3


"""
Power Control xAPP Agent
"""


"""
Resource Allocation xAPP Agent
"""
