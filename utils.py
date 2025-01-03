# -- Public Imports
import os
import numpy as np

# -- Private Imports

# -- Global Variables


# -- Functions

class ActionMapper:
    def __init__(self, minVal, maxVal):
        # Total number of discrete actions
        self.minVal = minVal
        self.maxVal = maxVal
        self.num_actions = (maxVal - minVal + 1)
        self.actions = list(range(minVal, maxVal+1))

    def idx_to_action(self, idx):
        """
        Map an index to a unique action
        """
        if idx < 0 or idx >= self.num_actions:
            raise ValueError(f"Index {idx} out of valid range [0, {self.num_actions - 1}]")

        return int(self.actions[idx])


def save_lists(file_path, ep_reward_list, ep_mean_reward_list, avg_reward_list, loss_by_iter_list):

    np.savetxt(os.path.join(file_path, r"ep_reward_list"), ep_reward_list)
    np.savetxt(os.path.join(file_path, r"ep_mean_reward_list"), ep_mean_reward_list)
    np.savetxt(os.path.join(file_path, r"avg_reward_list"), avg_reward_list)
    np.savetxt(os.path.join(file_path, r"loss_by_iter_list"), loss_by_iter_list)

    print(f"Successfully saved lists in {file_path}!!!")