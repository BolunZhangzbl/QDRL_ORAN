# Define eMBB Network Slice
class eMBB():
    """
    eMBB Network Slice aims to maximize the total throughput, it contains 3 UEs
    """
    def __init__(self, num_ues=3, num_rbs_allocated=0, slice_weight=0.3):
        self.slice_type = 'embb'

        self.num_ues = num_ues
        self.num_rbs_allocated = num_rbs_allocated

        # Traffic Flow
        self.packet_size = 32     # Bytes
        self.traffic_rate = 1000  # Constant traffic rate (bps) for each UE
        self.traffic_total = 0    # bps
        self.queue_total = 0      # bits
        self.service_rate = 2000  # Service rate per UE (bps)

        self.poisson_lambda = 1.0

        self.slice_weight = slice_weight

        # UEs
        self.UEs = [UE(self.packet_size) for _ in range(self.num_ues)]

    def update_traffic(self):
        """
        Update the traffic for each UE in the eMBB slice based on constant bit rate (CBR).
        The traffic rate for each UE is constant.
        """
        # Update traffic for each UE with constant rate (traffic_rate is fixed)
        for ue in self.UEs:
            packets_per_sec = np.random.poisson(self.poisson_lambda)

            traffic_size_bits = packets_per_sec * self.packet_size * bits_per_byte
            ue.add_to_queue(traffic_size_bits, self.service_rate)

        self.traffic_total = np.sum([ue.traffic_curr for ue in self.UEs])
        self.queue_total = np.sum([ue.get_queue_length() for ue in self.UEs])

    # def calculate_queue_delay(self):
    #     """
    #     Calculate the queue delay for each UE based on the traffic rate and service rate.
    #     """
    #     for ue in self.UEs:
    #         traffic_rate = ue.traffic_curr  # Arrival rate (bps)
    #         if traffic_rate < self.service_rate:
    #             delay = ue.traffic_curr / (self.service_rate - traffic_rate)  # Queue delay formula
    #         else:
    #             delay = float('inf')  # Infinite delay if service rate is insufficient
    #         ue.set_delay(delay)

    def assign_rbs_ppf(self, num_rbs_assign):
        pass

    def update_params(self):
        pass


# Define URLLC Network Slice
class URLLC():
    """
    URLLC Network Slice aims to minimize the average delay of packets
    """
    def __init__(self, num_ues=3, num_rbs_allocated=0, slice_weight=0.3):
        self.slice_type = 'urllc'

        self.num_ues = num_ues
        self.num_rbs_allocated = num_rbs_allocated

        # Traffic Flow
        self.packet_size = 16      # Bytes
        self.traffic_total = 0     # bps
        self.queue_total = 0       # bits
        self.poisson_lambda = 1.0

        # Bandwidth at each slice
        self.bandwidth_total = self.num_rbs_allocated * bandwidth_per_rb
        self.service_rate = self.bandwidth_total / self.num_ues

        self.slice_weight = slice_weight

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


    def assign_rbs_ppf(self, num_rbs_assign):
        pass

    def update_params(self):
        pass