import sys

from collections import defaultdict
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import traceback

from common.argparser import parse
import common.config as cfg
from common.proc_handler import cleanup, setup_graceful_exit
from common.utils import log_time, logger,\
    sep, time_stp, torch_device, weights_init
from data.data import get_loader
from models.autoencoder import get_autoencoder
from models.eigan import get_disrimininator
from models.pix2pix import define_G, define_D, GANLoss
from models.resnet import get_resnet


def main(
        expt,
        model_name,
        args
):
    device = torch_device(args.device, args.gpu_id[0])
    net_G = get_autoencoder(args.net_type)(
        cfg.flat_input_sizes[expt],
        args.encoding_dim).to(device)
    net_G = nn.DataParallel(net_G, device_ids=args.gpu_id)

    if args.load_w:
        print("Loading weights...\n{}\n{}".format(args.ckpt_g))
        net_G.load_state_dict(torch.load(args.ckpt_g))
    elif args.init_w:
        print("Init weights...")
        net_G.apply(weights_init)

    if args.optimizer == 'sgd':
        optim = torch.optim.SGD
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam
    opt_G = torch.optim.Adam(net_G.parameters(), lr=args.lr_g,
                             weight_decay=args.weight_decays[0])

    criterionGAN = nn.MSELoss().to(device)

    train_loader = get_loader(expt, args.batch_size, True,
                              img_size=args.img_size, subset=args.subset)
    valid_loader = get_loader(expt, args.test_batch_size,
                              False, img_size=args.img_size, subset=args.subset)

    loss_history = defaultdict(list)
    acc_history = defaultdict(list)
    for epoch in range(args.n_epochs):
        logging.info("Train Epoch \t Loss_G")

        for iteration, (image, labels) in enumerate(train_loader, 1):
            real = image.to(device)
            fake = net_G(real)

            opt_G.zero_grad()
            loss_g = criterionGAN(real.flatten(), fake.flatten())
            loss_g.backward()
            opt_G.step()
            lossg = loss_g.item()

            logging.info(
                '[{}]({}/{}) \t {:.4f}'.format(
                    epoch, iteration, len(train_loader),
                    lossg))

        loss_history['train_epoch'].append(epoch)
        loss_history['train_G'].append(lossg)
        acc_history['train_epoch'].append(epoch)

        logging.info("Valid Epoch \t Loss_G")

        loss_g_batch = 0
        for iteration, (image, labels) in enumerate(valid_loader, 1):

            real = image.to(device)
            fake = net_G(real)

            loss_g = criterionGAN(real.flatten(), fake.flatten())
            loss_g_batch += loss_g.item()
            lossg = loss_g.item()

        num_samples = len(valid_loader)
        logging.info(
            '[{}](batch) \t {:.4f}'.format(
                epoch,
                loss_g_batch / num_samples))

        loss_history['valid_epoch'].append(epoch)
        loss_history['valid_G'].append(loss_g_batch/num_samples)
        acc_history['valid_epoch'].append(epoch)

    train_loss_keys = [
        _ for _ in loss_history if 'train' in _ and 'epoch' not in _]
    valid_loss_keys = [
        _ for _ in loss_history if 'valid' in _ and 'epoch' not in _]
    train_acc_keys = [
        _ for _ in acc_history if 'train' in _ and 'epoch' not in _]
    valid_acc_keys = [
        _ for _ in acc_history if 'valid' in _ and 'epoch' not in _]

    cols = 5
    rows = len(train_loss_keys)//cols + 1
    fig = plt.figure(figsize=(7*cols, 5*rows))
    base = cols*100 + rows*10
    for idx, (tr_l, val_l) in enumerate(zip(train_loss_keys, valid_loss_keys)):
        ax = fig.add_subplot(rows, cols, idx+1)
        ax.plot(loss_history['train_epoch'], loss_history[tr_l], 'b.:')
        ax.plot(loss_history['valid_epoch'], loss_history[val_l], 'bs-.')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_title(tr_l[6:])
        ax.grid()
        if tr_l in acc_history:
            ax2 = plt.twinx()
            ax2.plot(acc_history['train_epoch'], acc_history[tr_l], 'r.:')
            ax2.plot(acc_history['valid_epoch'], acc_history[val_l], 'rs-.')
            ax2.set_ylabel('accuracy')
    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    plt_ckpt = '{}/{}/plots/{}.jpg'.format(
        cfg.ckpt_folder, expt, model_name)
    logging.info('Plot: {}'.format(plt_ckpt))
    plt.savefig(plt_ckpt, bbox_inches='tight', dpi=80)

    hist_ckpt = '{}/{}/history/{}.pkl'.format(
        cfg.ckpt_folder, expt, model_name)
    logging.info('History: {}'.format(hist_ckpt))
    pkl.dump((loss_history, acc_history), open(hist_ckpt, 'wb'))

    model_ckpt = '{}/{}/models/{}_g.stop'.format(
        cfg.ckpt_folder, expt, model_name)
    logging.info('Model: {}'.format(model_ckpt))
    torch.save(net_G.state_dict(), model_ckpt)


if __name__ == '__main__':
    setup_graceful_exit()
    args = parse()
    model = '{}_{}_dim_{}'.format(
        'pretrain_autoencoder', args.net_type, args.encoding_dim)
    pr_time, fl_time = time_stp()
    logger(args.expt, model)

    log_time('Start', pr_time)
    sep()
    logging.info(json.dumps(args.__dict__, indent=2))

    try:
        main(
            expt=args.expt,
            model_name=model,
            args=args
        )
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        cleanup()
