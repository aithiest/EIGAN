import joblib
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.utils.data as utils

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.argparser import comparison_argparse
from common.utility import log_shapes, log_time, torch_device,\
    time_stp, logger, sep, weights_init, load_processed_data
from common.torchsummary import summary
import matplotlib
from models.eigan import DiscriminatorFCN

matplotlib.rcParams.update({'font.size': 14})


def main(expt, model):
    pca_1 = pkl.load(open('checkpoints/mnist/ind_pca_training_history_01_30_2020_23_28_51.pkl', 'rb'))
    auto_1 = pkl.load(open('checkpoints/mnist/ind_autoencoder_training_history_01_30_2020_23_35_33.pkl', 'rb'))
    # dp_1 = pkl.load(open('checkpoints/mnist/ind_dp_training_history_02_01_2020_02_35_49.pkl', 'rb'))
    gan_1 = pkl.load(open('checkpoints/mnist/ind_gan_training_history_01_31_2020_16_05_44.pkl', 'rb'))

    u = pkl.load(open('checkpoints/mnist/eigan_training_history_02_05_2020_19_54_36_A_device_cuda_dim_1024_hidden_2048_batch_4096_epochs_501_lrencd_0.01_lrally_1e-05_lradvr_1e-05_tr_0.4023_val_1.7302.pkl', 'rb'))

    plt.figure()
    fig = plt.figure(figsize=(15, 4))
    ax1 = fig.add_subplot(131)
    ax3 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    t3, t1, t2 = '(b)', '(a)', '(c)'

    ax3.plot(pca_1['epoch']['valid'], gan_1['encoder']['advr_1_valid'], 'r')
    ax3.plot(pca_1['epoch']['valid'], auto_1['autoencoder']['advr_1_valid'], 'b')
    ax3.plot(pca_1['epoch']['valid'], pca_1['pca']['advr_1_valid'], 'g')
    # ax3.plot(pca_1['epoch']['valid'], dp_1['dp']['ally_valid'], 'y')
    ax3.legend([
        'EIGAN',
        'Autoencoder',
        'PCA',
        'DP',
    ],prop={'size':10})
    ax3.set_title(t3, y=-0.32)
    ax3.set_xlabel('epochs')
    ax3.set_ylabel('ally log loss')
    ax3.grid()
    ax3.text(320,1.68, 'Lower is better', fontsize=12, color='r')
    ax3.set_ylim(bottom=1.4)

    ax1.plot(pca_1['epoch']['valid'], gan_1['encoder']['ally_valid'], 'r--')
    ax1.plot(pca_1['epoch']['valid'], auto_1['autoencoder']['ally_valid'], 'b--')
    ax1.plot(pca_1['epoch']['valid'], pca_1['pca']['ally_valid'], 'g--')
    ax1.set_title(t1, y=-0.32)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('adversary log loss')
    ax1.grid()
    ax1.text(320,0.57, 'Higher is better', fontsize=12, color='r')
    ax1.set_ylim(bottom=0.5)

    ax2.plot(u[0], u[2], 'r', label='encoder loss')
    ax2.plot(np.nan, 'b', label = 'adversary loss')
    ax4 = ax2.twinx()
    ax4.plot(u[0], u[6], 'b')
    ax2.set_title('(c)', y=-0.32)
    ax2.legend(prop={'size':10})
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('encoder loss')
    ax2.grid()
    ax4.set_ylabel('adversary loss')

    fig.subplots_adjust(wspace=0.4)


    plot_location = 'plots/{}/{}_{}.png'.format(expt, 'all', model)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location, bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    expt = 'mnist'
    model = 'comparison'
    marker = 'A'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time, marker)

    log_time('Start', pr_time)
    main(expt, model)
    log_time('End', time_stp()[0])
    sep()
