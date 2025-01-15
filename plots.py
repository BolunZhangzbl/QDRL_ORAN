# -- Public Imports
import os
import numpy as np
import matplotlib.pyplot as plt


# -- Private Imports
from utils import *
from parameters import *


# -- Global Variables

dir_base = os.path.dirname(os.path.abspath(__file__))

dict_filename = dict(
    loss="loss_by_iter_list.txt",
    reward="ep_mean_reward_list.txt",
    avg_reward="avg_reward_list.txt"
)


dict_markers = {
    'qnn-irl': '^-',
    'qnn-frl': 's-',
    'dnn-irl': 'o--',
    'dnn-frl': 'D--'
}

dict_colors = {
    'qnn-irl': 'blue',
    'qnn-frl': 'green',
    'dnn-irl': 'red',
    'dnn-frl': 'orange'
}


# -- Functions

def plot_convergence(metric, str_lambda='3366', model_type='both'):

    assert metric in ('loss', 'reward', 'avg_reward')
    assert model_type in ('both', 'qnn', 'dnn')
    file_name = dict_filename.get(metric)

    if model_type == 'both':
        dict_file_path = {
            f"{model_type}-{key}": os.path.join(dir_base, "save_dqn", model_type, str_lambda, key, "save_lists", file_name) for
            model_type in ['qnn', 'dnn'] for key in ['frl', 'irl']}
    else:
        dict_file_path = {f"{model_type}-{key}": os.path.join(dir_base, "save_dqn", model_type, str_lambda, key, "save_lists", file_name) for key in ['frl', 'irl']}
    print(dict_file_path)

    dict_metric_list = {key: np.loadtxt(val) for key, val in dict_file_path.items()}
    dict_iter_range = {key: np.array(range(len(val))) for key, val in dict_metric_list.items()}

    plt.figure(figsize=(15, 10))
    for idx, (key, val) in enumerate(dict_metric_list.items()):
        val = smooth_curve(val, 50)
        plt.semilogy(dict_iter_range.get(key), val, dict_markers.get(key), color=dict_colors.get(key),
                     mfc='none', alpha=0.8, lw=2, markersize=3, label=key.upper())

    scale_type = 'linear' if 'reward' in metric else 'log'
    plt.yscale(scale_type)

    plt.xlabel('Iter', fontsize=30)
    plt.ylabel(f'{metric.upper()}', fontsize=30)
    # plt.xlim([200, 500])
    # plt.ylim([-10000, 10000])
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='best', fontsize=27)
    plt.grid(True, which='both', linestyle='--')
    plt.show()


plot_convergence(metric='reward', str_lambda='40408080', model_type='both')