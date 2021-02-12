from collections import defaultdict
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import traceback
import sys

from argparser import parse
import config as cfg
from data import get_loader
from utils import log_time, torch_device,\
    time_stp, logger, sep, weights_init
from pix2pix import define_G, define_D, GANLoss
from proc_handler import cleanup, setup_graceful_exit
from resnet import get_resnet
from torchsummary import summary

sys.path.append('./../')
from models.eigan import GeneratorLogistic, DiscriminatorLogistic  # noqa


def main(
        expt,
        model_name,
        device,
        gpu_id,
        train_nets,
        optimizer,
        n_classes,
        batch_size,
        test_batch_size,
        init_w,
        load_w,
        ckpt_g,
        ckpt_clfs,
        n_epochs,
        lr_g,
        lr_clfs,
        weight_decays,
        milestones,
        gamma,
):
    device = torch_device(device, gpu_id[0])
    num_clfs = len(n_classes)

    encoder = GeneratorLogistic(cfg.input_sizes[expt],
                                cfg.input_sizes[expt]).to(device)
    clfs = [DiscriminatorLogistic(cfg.input_sizes[expt], _).to(device)
            for _ in n_classes]

    if len(gpu_id) > 1:
        encoder = nn.DataParallel(encoder, device_ids=gpu_id)
        clfs = [nn.DataParallel(clf, device_ids=gpu_id) for clf in clfs]

    if load_w:
        print("Loading weights...\n{}".format(ckpt_g))
        encoder.load_state_dict(torch.load(ckpt_g))
        for clf, ckpt in zip(clfs, ckpt_clfs):
            print(ckpt)
            clf.load_state_dict(torch.load(ckpt))
    elif init_w:
        print("Init weights...")
        encoder.apply(weights_init)
        for clf in clfs:
            clf.apply(weights_init)

    if optimizer == 'sgd':
        optim = torch.optim.SGD
    elif optimizer == 'adam':
        optim = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.MultiStepLR
    opt_encoder = optim(encoder.parameters(), lr=lr_g, momentum=0.9,
                        weight_decay=weight_decays[0])
    opt_clfs = [optim(clf.parameters(), lr=lr, momentum=0.9,
                      weight_decay=weight_decays[1])
                for lr, clf in zip(lr_clfs, clfs)]
    sch_clfs = [scheduler(optim, milestones, gamma=gamma)
                for optim in opt_clfs]

    assert len(opt_clfs) == num_clfs

    criterionGAN = nn.MSELoss()
    criterionNLL = nn.CrossEntropyLoss().to(device)

    train_loader = get_loader(expt, batch_size, True)
    valid_loader = get_loader(expt, test_batch_size, False)

    loss_history = defaultdict(list)
    acc_history = defaultdict(list)
    for epoch in range(n_epochs):
        logging.info("Train Epoch \t Loss_E " +
                     ' '.join(["\t Clf: {}".format(_)
                               for _ in range(num_clfs)]))

        for iteration, (image, label1, label2) in enumerate(train_loader, 1):
            labels = [label1, label2]
            real = image.to(device)

            if 'gan' in train_nets:
                opt_encoder.zero_grad()
                fake = encoder(real)
                loss_e = criterionGAN(fake, real)
                loss_e.backward()
                opt_encoder.step()

                losse = loss_e.item()
            else:
                losse = 0

            if 'clf' in train_nets:
                X = image.to(device)
                ys = [_.long().flatten().to(device) for _ in labels]

                [opt.zero_grad() for opt in opt_clfs]
                ys_hat = [clf(X) for clf in clfs]
                loss = [criterionNLL(y_hat, y)
                        for y_hat, y in zip(ys_hat,
                                            ys)]
                ys_hat = [_.argmax(1, keepdim=True)
                          for _ in ys_hat]
                acc = [y_hat.eq(y.view_as(y_hat)).sum().item()/len(y)
                       for y_hat, y in zip(ys_hat, ys)]
                [l.backward() for l in loss]
                [opt.step() for opt in opt_clfs]

                iloss = [l.item() for l in loss]
                assert len(iloss) == num_clfs
            else:
                iloss = [0]*num_clfs
                acc = [0]*num_clfs

            logging.info(
                '[{}]({}/{}) \t {:.4f} '.format(
                    epoch, iteration, len(train_loader),
                    losse
                ) +
                ' '.join(['\t {:.4f} ({:.2f})'.format(
                    l, a) for l, a in zip(iloss, acc)])
            )

        loss_history['train_epoch'].append(epoch)
        loss_history['train_E'].append(losse)
        acc_history['train_epoch'].append(epoch)
        for idx, (l, a) in enumerate(zip(iloss, acc)):
            loss_history['train_M_{}'.format(idx)].append(l)
            acc_history['train_M_{}'.format(idx)].append(a)

        logging.info("Valid Epoch \t Loss_G " +
                     ' '.join(["\t Clf: {}".format(_) for _ in range(num_clfs)]))

        loss_e_batch = 0
        loss_m_batch = [0 for _ in range(num_clfs)]
        acc_m_batch = [0 for _ in range(num_clfs)]
        for iteration, (image, label1, label2) in enumerate(valid_loader, 1):
            labels = [label1, label2]
            if 'clf' in train_nets:
                X = image.to(device)
                ys = [_.long().flatten().to(device) for _ in labels]

                ys_hat = [clf(X) for clf in clfs]
                loss = [criterionNLL(y_hat, y)
                        for y_hat, y in zip(ys_hat, ys)]
                ys_hat = [_.argmax(1, keepdim=True)
                          for _ in ys_hat]
                acc = [y_hat.eq(y.view_as(y_hat)).sum().item()/len(y)
                       for y_hat, y in zip(ys_hat, ys)]

                iloss = [l.item() for l in loss]
                for idx, (l, a) in enumerate(zip(iloss, acc)):
                    loss_m_batch[idx] += l
                    acc_m_batch[idx] += a
            else:
                iloss = [0]*num_clfs
                acc = [0]*num_clfs

            if 'gan' in train_nets:
                real = image.to(device)
                fake = encoder(real)

                loss_e = criterionGAN(fake, real)
                losse = loss_e.item()
                loss_e_batch += losse
            else:
                losse = 0

            # logging.info(
            #     '[{}]({}/{}) \t {:.4f} \t {:.4f} '.format(
            #         epoch, iteration, len(valid_loader),
            #         lossd, lossg
            #     ) +
            #     ' '.join(['\t {:.4f} ({:.2f})'.format(
            #         l, a) for l, a in zip(iloss, acc)])
            # )

        num_samples = len(valid_loader)
        logging.info(
            '[{}](batch) \t {:.4f} '.format(
                epoch,
                loss_e_batch / num_samples
            ) +
            ' '.join(['\t {:.4f} ({:.2f})'.format(
                l/num_samples, a/num_samples) for l, a in zip(loss_m_batch, acc_m_batch)])
        )

        loss_history['valid_epoch'].append(epoch)
        loss_history['valid_E'].append(loss_e_batch/num_samples)
        acc_history['valid_epoch'].append(epoch)
        for idx, (l, a) in enumerate(zip(loss_m_batch, acc_m_batch)):
            loss_history['valid_M_{}'.format(idx)].append(l/num_samples)
            acc_history['valid_M_{}'.format(idx)].append(a/num_samples)

        [sch.step() for sch in sch_clfs]

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

    if 'gan' in train_nets:
        model_ckpt = '{}/{}/models/{}_e.stop'.format(
            cfg.ckpt_folder, expt, model_name)
        logging.info('Model: {}'.format(model_ckpt))
        torch.save(encoder.state_dict(), model_ckpt)

    if 'clf' in train_nets:
        for idx, clf in enumerate(clfs):
            model_ckpt = '{}/{}/models/{}_clf_{}.stop'.format(
                cfg.ckpt_folder, expt, model_name, idx)
            logging.info('Model: {}'.format(model_ckpt))
            torch.save(clf.state_dict(), model_ckpt)


if __name__ == '__main__':
    setup_graceful_exit()
    model = 'pretrain'
    args = parse()
    pr_time, fl_time = time_stp()
    logger(args.expt, model)

    log_time('Start', pr_time)
    sep()
    logging.info(json.dumps(args.__dict__, indent=2))

    try:
        main(
            expt=args.expt,
            model_name=model,
            device=args.device,
            gpu_id=args.gpu_id,
            train_nets=args.train_nets,
            optimizer=args.optimizer,
            n_classes=args.n_classes,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            init_w=args.init_w,
            load_w=args.load_w,
            ckpt_g=args.ckpt_g,
            ckpt_clfs=args.ckpt_clfs,
            n_epochs=args.n_epochs,
            lr_g=args.lr_g,
            lr_clfs=args.lr_clfs,
            weight_decays=args.weight_decays,
            milestones=args.milestones,
            gamma=args.gamma,
        )
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        cleanup()
