import logging
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import torch.utils.data as utils

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from common.argparser import autoencoder_argparse
from common.utility import log_time, torch_device,\
    time_stp, logger, sep
from common.torchsummary import summary

from preprocessing import get_data

from models.autoencoder import AutoEncoderBasic


def main(
        model,
        time_stamp,
        device,
        ally_classes,
        advr_classes,
        encoding_dim,
        test_size,
        batch_size,
        n_epochs,
        shuffle,
        lr,
        expt,
        ):

    device = torch_device(device=device)

    X_normalized_train, X_normalized_valid,\
        y_ally_train, y_ally_valid, \
        y_advr_1_train, y_advr_1_valid, \
        y_advr_2_train, y_advr_2_valid = get_data(expt, test_size)

    dataset_train = utils.TensorDataset(torch.Tensor(X_normalized_train))
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    dataset_valid = utils.TensorDataset(torch.Tensor(X_normalized_valid))
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=batch_size, shuffle=False, num_workers=2)

    auto_encoder = AutoEncoderBasic(
        input_size=X_normalized_train.shape[1],
        encoding_dim=encoding_dim
    ).to(device)

    criterion = torch.nn.MSELoss()
    adam_optim = torch.optim.Adam
    optimizer = adam_optim(auto_encoder.parameters(), lr=lr)

    summary(auto_encoder, input_size=(1, X_normalized_valid.shape[1]))

    h_epoch = []
    h_valid = []
    h_train = []

    auto_encoder.train()

    sep()
    logging.info("epoch \t Aencoder_train \t Aencoder_valid")

    for epoch in range(n_epochs):

        nsamples = 0
        iloss = 0
        for data in dataloader_train:
            optimizer.zero_grad()

            X_torch = data[0].to(device)
            X_torch_hat = auto_encoder(X_torch)
            loss = criterion(X_torch_hat, X_torch)
            loss.backward()
            optimizer.step()

            nsamples += 1
            iloss += loss.item()

        if epoch % int(n_epochs/10) != 0:
            continue

        h_epoch.append(epoch)
        h_train.append(iloss/nsamples)

        nsamples = 0
        iloss = 0
        for data in dataloader_valid:
            X_torch = data[0].to(device)
            X_torch_hat = auto_encoder(X_torch)
            loss = criterion(X_torch_hat, X_torch)

            nsamples += 1
            iloss += loss.item()
        h_valid.append(iloss/nsamples)

        logging.info('{} \t {:.8f} \t {:.8f}'.format(
            h_epoch[-1],
            h_train[-1],
            h_valid[-1],
        ))

    config_summary = 'device_{}_dim_{}_batch_{}_epochs_{}_lr_{}_tr_{:.4f}_val_{:.4f}'\
        .format(
            device,
            encoding_dim,
            batch_size,
            n_epochs,
            lr,
            h_train[-1],
            h_valid[-1],
        )

    plt.plot(h_epoch, h_train, 'r--')
    plt.plot(h_epoch, h_valid, 'b--')
    plt.legend(['train_loss', 'valid_loss'])
    plt.title("autoencoder training {}".format(config_summary))

    plot_location = 'plots/{}/{}_training_{}_{}.png'.format(
        expt, model, time_stamp, config_summary)
    sep()
    logging.info('Saving: {}'.format(plot_location))
    plt.savefig(plot_location)
    checkpoint_location = \
        'checkpoints/{}/{}_training_history_{}_{}.pkl'.format(
            expt, model, time_stamp, config_summary)
    logging.info('Saving: {}'.format(checkpoint_location))
    pkl.dump((h_epoch, h_train, h_valid), open(checkpoint_location, 'wb'))

    model_ckpt = 'checkpoints/{}/{}_torch_model_{}_{}.pkl'.format(
            expt, model, time_stamp, config_summary)
    logging.info('Saving: {}'.format(model_ckpt))
    torch.save(auto_encoder, model_ckpt)


if __name__ == "__main__":
    expt = 'mnist'
    model = 'autoencoder_basic'
    marker = 'A'
    pr_time, fl_time = time_stp()

    logger(expt, model, fl_time, marker)

    log_time('Start', pr_time)
    args = autoencoder_argparse()
    main(
        model=model,
        time_stamp=fl_time,
        device=args['device'],
        ally_classes=args['n_ally'],
        advr_classes=args['n_advr'],
        encoding_dim=args['dim'],
        test_size=args['test_size'],
        batch_size=args['batch_size'],
        n_epochs=args['n_epochs'],
        shuffle=args['shuffle'] == 1,
        lr=args['lr'],
        expt=args['expt'],
    )
    log_time('End', time_stp()[0])
    sep()
