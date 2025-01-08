# -- Public Imports
import os
import numpy as np
import matplotlib.pyplot as plt


# -- Private Imports
from parameters import *


# -- Global Variables

dir_base = os.path.dirname(os.path.abspath(__file__))

dict_filename = dict(
    loss="loss_by_iter_list.txt",
    reward="ep_mean_reward_list.txt",
    avg_reward="avg_reward_list.txt"
)

dict_markers = dict(
    irl='^-',
    frl='o--',
)

dict_colors = dict(
    irl='blue',
    frl='red',
)


# -- Functions

def plot_convergence(metric, str_lambda='3366'):

    assert metric in ('loss', 'reward', 'avg_reward')
    file_name = dict_filename.get(metric)

    dict_file_path = {key: os.path.join(dir_base, "save_dqn", str_lambda, key, "save_lists", file_name) for key in ['frl', 'irl']}
    dict_metric_list = {key: np.loadtxt(val) for key, val in dict_file_path.items()}
    print(dict_metric_list)
    dict_iter_range = {key: np.array(range(len(val))) for key, val in dict_metric_list.items()}

    plt.figure(figsize=(15, 10))
    for idx, (key, val) in enumerate(dict_metric_list.items()):
        plt.semilogy(dict_iter_range.get(key), val, dict_markers.get(key), color=dict_colors.get(key),
                     mfc='none', alpha=0.8, lw=2, markersize=3, label=key.upper())

    # plt.xlim([-10, 96])
    # plt.ylim([-2000000.0, 0])

    scale_type = 'linear' if 'reward' in metric else 'log'
    plt.yscale(scale_type)

    plt.xlabel('Iter', fontsize=30)
    plt.ylabel(f'{metric.upper()}', fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='best', fontsize=27)
    plt.grid(True, which='both', linestyle='--')
    plt.show()


plot_convergence(metric='loss', str_lambda=''.join(str(val) for val in dict_poisson_lambda.values()))