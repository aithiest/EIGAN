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
from common.utils import get_network, log_time, torch_device,\
    time_stp, logger, sep, weights_init
from data.data import get_loader
from models.autoencoder import get_autoencoder
from models.eigan import get_disrimininator
from models.pix2pix import define_G
from models.resnet import get_resnet


def main(
        expt,
        model_name,
        args
):
    device = torch_device(args.device, args.gpu_id[0])
    num_clfs = len([_ for _ in args.n_classes if _ > 0])
    if args.arch == 'resnet':
        print('Using resnet')
        Net = get_resnet(args.num_layers)
    elif args.arch in ['linear', 'logistic', 'fcn']:
        print('Using {}'.format(args.arch))
        Net = get_disrimininator(args.arch)
    else:
        print('Using {}'.format(args.arch))
        Net = get_network(args.arch, args.num_layers)

    if args.net_type in ['linear', 'logistic', 'fcn']:
        net_G = get_autoencoder(args.net_type)(
            cfg.flat_input_sizes[expt],
            args.encoding_dim).to(device)
    else:
        net_G = define_G(cfg.num_channels[expt],
                         cfg.num_channels[expt],
                         64, gpu_id=device)

    if args.net_type in ['linear', 'logistic', 'fcn']:
        clfs = [Net(cfg.flat_input_sizes[expt], _).to(device)
                for _ in args.n_classes]
    elif args.net_type == 'deep':
        clfs = [Net(num_channels=cfg.num_channels[expt],
                    num_classes=_).to(device) for _ in args.n_classes if _ > 0]
    clfs = [nn.DataParallel(clf, device_ids=args.gpu_id)
            for clf in clfs]
    net_G = nn.DataParallel(net_G, device_ids=args.gpu_id)

    assert len(clfs) == num_clfs

    print("Loading weights...\n{}".format(args.ckpt_g))
    net_G.load_state_dict(torch.load(args.ckpt_g))
    if args.init_w:
        print("Init weights...")
        for clf in clfs:
            clf.apply(weights_init)

    scheduler = torch.optim.lr_scheduler.MultiStepLR
    if args.optimizer == 'sgd':
        opt_clfs = [torch.optim.SGD(clf.parameters(), lr=lr,  momentum=0.9,
                                    weight_decay=args.weight_decays[0])
                    for lr, clf in zip(args.lr_clfs, clfs)]
    elif args.optimizer == 'adam':
        opt_clfs = [torch.optim.SGD(clf.parameters(), lr=lr,  weight_decay=args.weight_decays[0])
                    for lr, clf in zip(args.lr_clfs, clfs)]
    sch_clfs = [scheduler(optim, args.milestones, gamma=args.gamma)
                for optim in opt_clfs]

    assert len(opt_clfs) == num_clfs

    criterionNLL = nn.CrossEntropyLoss().to(device)

    train_loader = get_loader(expt, args.batch_size, True,
                              img_size=args.img_size, subset=args.subset)
    valid_loader = get_loader(expt, args.test_batch_size,
                              False, img_size=args.img_size, subset=args.subset)

    loss_history = defaultdict(list)
    acc_history = defaultdict(list)
    for epoch in range(args.n_epochs):
        logging.info("Train Epoch " + ' '.join(["\t Clf: {}".format(_)
                                                for _ in range(num_clfs)]))

        for iteration, (image, labels) in enumerate(train_loader, 1):
            real = image.to(device)

            with torch.no_grad():
                X = net_G(real)
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

            X = net_G(image.to(device))
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

            logging.info(
                '[{}]({}/{}) '.format(
                    epoch, iteration, len(valid_loader),
                ) +
                ' '.join(['\t {:.4f} ({:.2f})'.format(
                    l, a) for l, a in zip(iloss, acc)])
            )

        num_samples = len(valid_loader)
        logging.info(
            '[{}](batch) '.format(
                epoch,
            ) +
            ' '.join(['\t {:.4f} ({:.2f})'.format(
                l/num_samples, a/num_samples) for l, a in zip(loss_m_batch, acc_m_batch)])
        )

        num_samples = len(valid_loader)
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
        ax.plot(loss_history['train_epoch'], loss_history[tr_l], 'b.:')
        ax.plot(loss_history['valid_epoch'], loss_history[val_l], 'bs-.')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_title(tr_l[6:])
        ax.grid()
        if tr_l in acc_history:
            logging.info('train: {} {}'.format(tr_l, max(acc_history[tr_l])))
            logging.info('val: {} {}'.format(val_l, max(acc_history[val_l])))
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

    for idx, clf in enumerate(clfs):
        model_ckpt = '{}/{}/models/{}_clf_{}.stop'.format(
            cfg.ckpt_folder, expt, model_name, idx)
        logging.info('Model: {}'.format(model_ckpt))
        torch.save(clf.state_dict(), model_ckpt)


if __name__ == '__main__':
    setup_graceful_exit()
    model = 'check_train'
    args = parse()
    arch = args.arch if not args.arch == 'resnet' else '{}{}'.format(
        args.arch, args.num_layers)
    model = '{}_{}_{}_{}'.format(
        model, arch, args.net_type, args.ckpt_g.split('/')[-1])
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
