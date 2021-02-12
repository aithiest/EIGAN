import sys
import pickle as pkl
import traceback

from collections import defaultdict
import json
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


from common.argparser import parse
import common.config as cfg
from common.proc_handler import cleanup, setup_graceful_exit
from common.utils import log_time, logger,\
    sep, time_stp, torch_device, weights_init
from data.data import get_loader
from models.eigan import get_disrimininator
from models.resnet import get_resnet


def main(
        expt,
        model_name,
        args
):
    device = torch_device(args.device, args.gpu_id[0])
    num_clfs = len(args.n_classes)
    if args.net_type in ['linear', 'logistic', 'fcn']:
        Net = get_disrimininator(args.net_type)
    elif args.net_type == 'deep':
        Net = get_resnet(args.num_layers)

    if args.net_type in ['linear', 'logistic', 'fcn']:
        clfs = [Net(cfg.flat_input_sizes[expt], _).to(device)
                for _ in args.n_classes]
    elif args.net_type == 'deep':
        clfs = [Net(num_channels=cfg.num_channels[expt],
                    num_classes=_).to(device) for _ in args.n_classes]
    clfs = [nn.DataParallel(clf, device_ids=args.gpu_id)
            for clf in clfs]

    assert len(clfs) == num_clfs

    if args.load_w:
        print("Loading weights...\n{}\n{}".format(args.ckpt_g, args.ckpt_d))
        for clf, ckpt in zip(clfs, args.ckpts_clfs):
            print(ckpt)
            clf.load_state_dict(torch.load(ckpt))
    elif args.init_w:
        print("Init weights...")
        for clf in clfs:
            clf.apply(weights_init)

    if args.optimizer == 'sgd':
        optim = torch.optim.SGD
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.MultiStepLR
    opt_clfs = [optim(clf.parameters(), lr=lr,
                      momentum=0.9, weight_decay=args. weight_decays[0])
                for lr, clf in zip(args.lr_clfs, clfs)]
    sch_clfs = [scheduler(optim, args.milestones, gamma=args.gamma)
                for optim in opt_clfs]

    assert len(opt_clfs) == num_clfs

    criterionL1 = nn.L1Loss().to(device)
    criterionNLL = nn.CrossEntropyLoss().to(device)

    train_loader = get_loader(expt, args.batch_size, True,
                              img_size=args.img_size, subset=args.subset)
    valid_loader = get_loader(expt, args.test_batch_size,
                              False, img_size=args.img_size, subset=args.subset)

    loss_history = defaultdict(list)
    acc_history = defaultdict(list)
    for epoch in range(args.n_epochs):
        logging.info("Train Epoch " +
                     ' '.join(["\t Clf: {}".format(_)
                               for _ in range(num_clfs)]))

        for iteration, (image, labels) in enumerate(train_loader, 1):
            X = image.to(device)
            ys = [_.flatten().to(device) for _ in labels]

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

            logging.info(
                '[{}]({}/{}) '.format(
                    epoch, iteration, len(train_loader),
                ) +
                ' '.join(['\t {:.4f} ({:.2f})'.format(
                    l, a) for l, a in zip(iloss, acc)])
            )

        loss_history['train_epoch'].append(epoch)
        acc_history['train_epoch'].append(epoch)
        for idx, (l, a) in enumerate(zip(iloss, acc)):
            loss_history['train_M_{}'.format(idx)].append(l)
            acc_history['train_M_{}'.format(idx)].append(a)

        logging.info("Valid Epoch " +
                     ' '.join(["\t Clf: {}".format(_) for _ in range(num_clfs)]))

        loss_m_batch = [0 for _ in range(num_clfs)]
        acc_m_batch = [0 for _ in range(num_clfs)]
        for iteration, (image, labels) in enumerate(valid_loader, 1):

            X = image.to(device)
            ys = [_.flatten().to(device) for _ in labels]

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

        num_samples = len(valid_loader)
        logging.info(
            '[{}](batch) '.format(
                epoch,
            ) +
            ' '.join(['\t {:.4f} ({:.2f})'.format(
                l/num_samples, a/num_samples) for l, a in zip(loss_m_batch, acc_m_batch)])
        )

        loss_history['valid_epoch'].append(epoch)
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
        ax.plot(loss_history['train_epoch'], loss_history[tr_l],
                'bx-', alpha=0.3)
        ax.plot(loss_history['valid_epoch'], loss_history[val_l], 'bs-')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_title(tr_l[6:])
        ax.grid()
        if tr_l in acc_history:
            logging.info('train: {} {}'.format(tr_l, max(acc_history[tr_l])))
            logging.info('val: {} {}'.format(val_l, max(acc_history[val_l])))
            ax2 = plt.twinx()
            ax2.plot(acc_history['train_epoch'], acc_history[tr_l],
                     'rx-', alpha=0.3)
            ax2.plot(acc_history['valid_epoch'], acc_history[val_l], 'rs-')
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

    for idx, clf in enumerate(clfs):
        model_ckpt = '{}/{}/models/{}_clf_{}.stop'.format(
            cfg.ckpt_folder, expt, model_name, idx)
        logging.info('Model: {}'.format(model_ckpt))
        torch.save(clf.state_dict(), model_ckpt)


if __name__ == '__main__':
    setup_graceful_exit()
    args = parse()
    model = '{}_{}'.format(
        'pretrain_clf', args.net_type)
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
