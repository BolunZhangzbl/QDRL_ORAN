# -- Public Imports
from gym import Env, spaces
import random
import itertools
import numpy as np

# -- Private Imports
from parameters import *

# -- Global Variables


# -- Functions

class SlicingEnv(Env):

    def __init__(self):
        self.oran = ORAN()

    def get_state_info(self, kth, nth):

        Hn = self.oran.BSs[kth].slices[nth].queue_total
        dm = np.sum([self.oran.get_total_delay(kth, nth, m) for m in range(3)])
        Pk = self.oran.BSs[kth].bs_power

        state_info = [Hn, dm, Pk]

        return state_info

    def step(self, actions):
        action_power = actions[0]
        action_resource = actions[1]

        # 1. Perform the action at ORAN
        # action.shape == (2,1)
        self.oran.set_power(action_power)

        # action.shape == (2,4)
        self.oran.set_rbs(action_resource)

        # 2. Calculate the reward
        reward_resource = self.oran.get_total_reward()
        reward_power = reward_resource - alpha * action_power[0] - alpha * action_power[1]
        self.rewards = [reward_power, reward_resource]

        if reward_power >= dict_reward_done.get('power') and reward_resource >= dict_reward_done.get('resource'):
            self.done = True

        # 3. Get the current state info after performing action
        # self.next_state.shape == (2,4,3)
        self.next_state = np.array([[self.get_state_info(k, n) for n in range(4)] for k in range(2)])

        return self.next_state, self.rewards, self.done

    def reset(self):
        # self.state.shape == (2,4,3)
        self.state = np.array([[self.get_state_info(k, n) for n in range(4)] for k in range(2)])
        self.rewards = [0, 0]
        self.done = False

        return self.state


# Define Resource Block class
class RB:
    """
    Resource Block contains info
    """
    def __init__(self, rth, kth=-1, nth=-1, mth=-1, is_allocated=False, power=0):
        self.rth = rth   # the idx of the RB
        self.kth = kth   # the idx of the BS that the RB is allocated
        self.nth = nth   # the idx of the slice that the RB is allocated
        self.mth = mth   # the idx of the UE that the RB is allocated
        self.is_allocated = is_allocated
        self.power = power

    def update_params(self, kth, nth, mth, is_allocated, power):
        self.kth = kth
        self.nth = nth
        self.mth = mth
        self.is_allocated = is_allocated
        self.power = power

    def reset(self):
        self.kth = -1
        self.nth = -1
        self.mth = -1
        self.is_allocated = False


# Define O-RAN class
class ORAN:
    """
    Open RAN Class which contains two BSs
    """
    def __init__(self):

        # Define BS
        self.num_gnbs = num_gnbs  # Number of BSs
        self.BSs = [BS(0, 0) for _ in range(2)]

        # Define hyper-parameters
        self.tx_power_max = P_max
        self.tx_power_min = -P_max
        self.BSs[0].bs_power = np.random.uniform(self.tx_power_min, self.tx_power_max)
        self.BSs[1].bs_power = np.random.uniform(self.tx_power_min, self.tx_power_max)
        self.Br = bandwidth_per_rb  # Bandwidth per RB (Hz)
        self.N0 = N0  # Noise Power Density (dBm/Hz)

        # Define RBs
        self.RBs = [RB(rth=idx) for idx in range(num_rbs)]

    def __str__(self):
        return 0

    def get_channel_gain(self):
        # Path Loss
        distance_bs2ue = np.random.uniform(d_min, d_max)
        pl = 128.1 + 37.6 * np.log10(distance_bs2ue)

        # Shadowing
        s = np.random.normal(0, 8)   # 0 mean, 8 dB std

        # channel gain
        gkm = tx_rx_ant_gain*2 - (pl + s)

        return gkm

    def get_sinr(self, kth, rth):
        assert kth in (0, 1)
        assert rth in range(100)

        if self.RBs[rth].kth == kth and self.RBs[rth].is_allocated == 1:
            gkm_num = self.get_channel_gain()
            numerator = self.RBs[rth].is_allocated * gkm_num * self.RBs[rth].power
        else:
            return 0

        denominator = 0
        for rth_ in range(num_rbs):
            if self.RBs[rth_].kth != kth and self.RBs[rth_].is_allocated == 1:
                gkm_den = self.get_channel_gain()
                denominator += 1 * gkm_den * self.RBs[rth].power + self.Br*self.N0

        sinr = numerator / denominator

        return sinr

    def get_link_capacity(self, kth):
        Ckm = 0
        for rth in range(num_rbs):
            Ckm += self.Br * np.log2(1 + self.get_sinr(kth, rth))

        return Ckm

    def get_tx_delay(self, kth, nth, mth):
        # nth {0, 1, 2, 3}      Idx of Slice
        # mth {0, 1, 2}         Idx of mobile devices

        Ckm = self.get_link_capacity(kth)
        Lm = self.BSs[kth].slices[nth].UEs[mth].traffic_curr

        tx_delay = Lm / Ckm

        return tx_delay

    def get_total_delay(self, kth, nth, mth):
        # nth {0, 1, 2, 3}      Idx of Slice
        # mth {0, 1, 2}         Idx of mobile devices

        tx_delay = self.get_tx_delay(kth, nth, mth)
        queue_delay = self.BSs[kth].slices[nth].UEs[mth].delay
        tx_re_delay = 4*TTI
        total_delay = tx_delay + queue_delay + tx_re_delay

        return total_delay

    def intra_slice_allocate_ppf(self, kth, nth, sum_rbs):
        """
        :return: the idx of the UE for PPF allocation
        """
        # nth {0, 1, 2, 3}      Idx of Slice
        # mth {0, 1, 2}         Idx of mobile devices
        assert isinstance(sum_rbs, int)

        Ckm = self.get_link_capacity(kth)
        UEs = self.BSs[kth].slices[nth].UEs

        tx_rates = [(ue.traffic_curr - ue.traffic_past) for ue in UEs]
        tx_rates = [elem if elem!=0 else float('inf') for elem in tx_rates]
        ppf_dist = [Ckm/tx_rate for tx_rate in tx_rates]
        ppf_dist_total = sum(ppf_dist)

        rbs_dist_intra = [round(sum_rbs * dist / ppf_dist_total) for dist in ppf_dist]

        # Adjust if there is a rounding issue (e.g., the sum of result isn't equal to integer)
        diff = sum_rbs - sum(rbs_dist_intra)
        for i in range(diff):
            rbs_dist_intra[i % len(rbs_dist_intra)] += 1

        return rbs_dist_intra

    def set_rbs(self, rbs_dist):

        # rbs_dist is the action performed by the Resource Allocation Agent
        # rbs_dist is the number of rbs distributions across 4 slices within a BS
        # rbs_dist.shape == (2, 4)

        # Assignment Loop
        for k in range(num_gnbs):
            # 1. Allocate the RBs to the BS
            sum_rbs = sum(rbs_dist[k])
            self.BSs[k].num_rbs_allocated = sum_rbs

            for n in range(len(rbs_dist)):
                # 2. Allocate the RBs to the slice
                self.BSs[k].slices[n].num_rbs_allocated = rbs_dist[k][n]

                # 3. Intra-slice RBs allocation to UEs using PPF
                rbs_dist_intra = self.intra_slice_allocate_ppf(k, n, rbs_dist[k][n])
                for m in range(len(rbs_dist_intra)):
                    self.BSs[k].slices[n].UEs[m].num_rbs_allocated = rbs_dist_intra[m]

                    # 4. Update info in the self.RBs
                    # 4.1 Find the indices of available RBs
                    indices_zero = list(itertools.islice((rb.rth for rb in self.RBs if rb.is_allocated==False), 3))
                    # 4.2 Update self.RBs
                    for r in indices_zero:
                        self.RBs[r].kth = k
                        self.RBs[r].nth = n
                        self.RBs[r].mth = m
                        self.RBs[r].is_allocated = True
                    # 4.3 Add the indices of RBs to UE for easier release
                        self.BSs[k].slices[n].UEs[m].rbs_indices.append(r)

    def set_power(self, a):
        # a.shape == (2, 1)

        for k in range(self.num_gnbs):
            self.BSs[k].bs_power = a[k]

            indices = [rb.rth for rb in self.RBs if rb.is_allocated == 1 and rb.kth == k]

            # Ensure the Tx Power is uniformly distributed across all the RBs
            a_uniform = a / len(indices)
            for idx in indices:
                self.RBs[idx].power = a_uniform

    def get_slice_reward(self, kth, nth):
        slice = self.BSs[kth].slices[nth]
        if slice.queue_total != 0:
            if slice.slice_type.startswith('embb'):
                reward = np.arctan(self.BSs[kth].slices[nth].traffic_total)
            else:
                reward = 1 - np.sum([self.get_total_delay(kth, nth, m) for m in range(3)])
        else:
            reward = 0

        # Multiply by its weight
        reward *= slice.slice_weight

        return reward

    def get_total_reward(self):
        reward_weighted_sum = 0
        for k in range(num_gnbs):
            for n in range(4):
                reward_weighted_sum += self.get_slice_reward(k, n)

        return reward_weighted_sum

    def clear_ue_rbs(self):
        for k in range(self.num_gnbs):
            for n in range(4):
                for m in range(3):
                    ue = self.BSs[k].slices[n].UEs[m]
                    if not ue.queue:
                        for r in ue.rbs_indices:
                            self.RBs[r].reset()


# Define Base Station
class BS:
    """
    Base Station Class which contains two eMBB slices and two URLLC slices
    """
    def __init__(self, bs_power=0, num_rbs_allocated=0):

        self.slices = [Slice(key) for key in dict_slice_weights.keys()]
        self.bs_power = bs_power
        self.num_rbs_allocated = num_rbs_allocated


# Define the general class for Slice
class Slice:
    """
    General Class for Network Slice: eMBB or URLLC
    """
    def __init__(self, slice_type, num_ues=3, num_rbs_allocated=0):

        assert slice_type in dict_slice_weights.keys()
        self.slice_type = slice_type
        self.slice_weight = dict_slice_weights.get(slice_type)

        self.num_ues = num_ues
        self.num_rbs_allocated = num_rbs_allocated

        # Traffic Flow
        self.packet_size = dict_packet_size.get(slice_type)  # Bytes
        self.traffic_rate = 1000  # Constant traffic rate (bps) for each UE
        self.traffic_total = 0  # bps
        self.queue_total = 0  # bits
        self.service_rate = 2000  # Service rate per UE (bps)

        self.poisson_lambda = dict_poisson_lambda.get(slice_type)

        # UEs
        self.UEs = [UE(self.packet_size) for _ in range(self.num_ues)]

    def update_traffic(self):
        """
        Simulate the traffic demands of each UE in the slice based on Poisson distribution.
        Each UE generates packets with the given packet size.
        :return:
        """

        for ue in self.UEs:
            packets_per_sec = np.random.poisson(self.poisson_lambda)

            traffic_size_bits = packets_per_sec * self.packet_size * bits_per_byte
            ue.add_to_queue(traffic_size_bits, self.service_rate)

        self.traffic_total = np.sum([ue.traffic_curr for ue in self.UEs])
        self.queue_total = np.sum([ue.get_queue_length() for ue in self.UEs])

        return self.traffic_total, self.queue_total


class UE:
    """
    User Equipment Class
    """
    def __init__(self, packet_size, num_rbs_allocated=0, max_queue_length=10):

        assert packet_size in (16, 32)

        self.traffic_curr = 0   # Current traffic in bps
        self.traffic_past = 0   # Traffic in the previous cycle (for delay calculation)
        self.delay = 0          # Total delay for the current packet (sec)
        self.queue = []         # FIFO queue for packets
        self.packet_size = packet_size             # bytes
        self.num_rbs_allocated = num_rbs_allocated
        self.rbs_indices = []
        self.max_queue_length = max_queue_length   # Maximum queue length (packets)

    def add_to_queue(self, traffic_size_bits, service_rate):
        # Convert traffic size in bits to number of packets
        traffic_size_packets = traffic_size_bits / (self.packet_size * bits_per_byte)

        # Add new packets to the queue
        for _ in range(int(traffic_size_packets)):
            if len(self.queue) < self.max_queue_length:
                self.queue.append(1)  # Add a packet to the queue

        # Calculate the number of packets that can be served in this cycle
        packets_served = int(service_rate / (self.packet_size * bits_per_byte))
        self.queue = self.queue[packets_served:]  # Serve packets, removing from the queue

        # Update current traffic (the size of the remaining packets in the queue)
        self.traffic_past = self.traffic_curr
        self.traffic_curr = len(self.queue) * (self.packet_size * bits_per_byte)  # Remaining traffic in queue

        # Calculate the queue delay based on the number of packets in the queue
        self.delay = self.calculate_queue_delay(service_rate)

    def calculate_queue_delay(self, service_rate):
        """
        Calculate the queue delay using the formula:
        queue_delay = (number_of_packets_in_queue) / (service_rate)
        where service_rate is in bits per second, and the queue length is in bits.
        """
        queue_length_bits = len(self.queue) * self.packet_size * bits_per_byte
        queue_delay = queue_length_bits / service_rate  # Time to serve the current queue in seconds
        return queue_delay

    def get_queue_packets(self):
        """Return the total number of queue packets"""
        return len(self.queue)

    def get_queue_length(self):
        """Return the total queue length in bits"""
        return len(self.queue) * self.packet_size * bits_per_byte

    def reset(self):
        """Reset the UE state"""
        self.traffic_curr = 0
        self.traffic_past = 0
        self.delay = 0
        self.queue = []