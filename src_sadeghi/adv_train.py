import sys
import pickle as pkl
import traceback

from collections import defaultdict
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common.argparser import parse
import common.config as cfg
from common.proc_handler import cleanup, setup_graceful_exit
from common.utils import log_time, torch_device,\
    time_stp, logger, sep, weights_init, eigan_loss
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
    num_clfs = len([_ for _ in args.n_classes if _ > 0])
    if args.net_type in ['linear', 'logistic', 'fcn']:
        Net = get_disrimininator(args.net_type)
    elif args.net_type == 'deep':
        Net = get_resnet(args.num_layers)

    if args.net_type in ['linear', 'logistic', 'fcn']:
        net_G = get_autoencoder(args.net_type)(
            cfg.flat_input_sizes[expt],
            args.encoding_dim).to(device)
    elif args.net_type == 'deep':
        net_G = define_G(cfg.num_channels[expt],
                         cfg.num_channels[expt],
                         64, gpu_id=device)
    if args.net_type in ['linear', 'logistic', 'fcn']:
        clfs = [Net(cfg.flat_input_sizes[expt], _).to(device)
                for _ in args.n_classes]
    elif args.net_type == 'deep':
        clfs = [Net(num_channels=cfg.num_channels[expt],
                    num_classes=_).to(device) for _ in args.n_classes]

    assert len(clfs) == num_clfs

    net_G = nn.DataParallel(net_G, device_ids=args.gpu_id)

    if args.load_w:
        print("Loading weights...\n{}".format(args.ckpt_g))
        net_G.load_state_dict(torch.load(args.ckpt_g))
        for clf, ckpt in zip(clfs, args.ckpt_clfs):
            print(ckpt)
            clf.load_state_dict(torch.load(ckpt))
    elif args.init_w:
        print("Init weights...")
        net_G.apply(weights_init)
        for clf in clfs:
            clf.apply(weights_init)

    clfs = [nn.DataParallel(clf, device_ids=args.gpu_id) for clf in clfs]

    if args.optimizer == 'sgd':
        optim = torch.optim.SGD
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.MultiStepLR
    opt_G = torch.optim.Adam(net_G.parameters(), lr=args.lr_g,
                             weight_decay=args.weight_decays[0])
    opt_clfs = [optim(clf.parameters(), lr=lr, weight_decay=args.weight_decays[1])
                for lr, clf in zip(args.lr_clfs, clfs)]
    sch_clfs = [scheduler(optim, args.milestones, gamma=args.gamma)
                for optim in opt_clfs]

    assert len(opt_clfs) == num_clfs

    criterionGAN = eigan_loss
    criterionNLL = nn.CrossEntropyLoss().to(device)

    train_loader = get_loader(expt, args.batch_size, True,
                              img_size=args.img_size, subset=args.subset)
    valid_loader = get_loader(expt, args.test_batch_size,
                              False, img_size=args.img_size, subset=args.subset)

    loss_history = defaultdict(list)
    acc_history = defaultdict(list)
    for epoch in range(args.n_epochs):
        logging.info("Train Epoch \t Loss_G " +
                     ' '.join(["\t Clf: {}".format(_)
                               for _ in range(num_clfs)]))

        for iteration, (image, labels) in enumerate(train_loader, 1):
            real = image.to(device)
            ys = [_.flatten().to(device)
                  for _, num_c in zip(labels, args.n_classes) if num_c > 0]

            with torch.no_grad():
                X = net_G(real)

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

            X = net_G(real)
            ys_hat = [clf(X) for clf in clfs]
            loss = [criterionNLL(y_hat, y)
                    for y_hat, y in zip(ys_hat,
                                        ys)]

            opt_G.zero_grad()
            loss_g = criterionGAN(loss, args.ei_array, args.n_classes)
            loss_g.backward()
            opt_G.step()

            logging.info(
                '[{}]({}/{}) \t {:.4f} '.format(
                    epoch, iteration, len(train_loader),
                    loss_g.item()
                ) +
                ' '.join(['\t {:.4f} ({:.2f})'.format(
                    l, a) for l, a in zip(iloss, acc)])
            )

        loss_history['train_epoch'].append(epoch)
        loss_history['train_G'].append(loss_g.item())
        acc_history['train_epoch'].append(epoch)
        for idx, (l, a) in enumerate(zip(iloss, acc)):
            loss_history['train_M_{}'.format(idx)].append(l)
            acc_history['train_M_{}'.format(idx)].append(a)

        logging.info("Valid Epoch \t Loss_G " +
                     ' '.join(["\t Clf: {}".format(_) for _ in range(num_clfs)]))

        loss_g_batch = 0
        loss_m_batch = [0 for _ in range(num_clfs)]
        acc_m_batch = [0 for _ in range(num_clfs)]
        for iteration, (image, labels) in enumerate(valid_loader, 1):

            real = image.to(device)
            fake = net_G(real)
            ys = [_.flatten().to(device)
                  for _, num_c in zip(labels, args.n_classes) if num_c > 0]

            ys_hat = [clf(fake) for clf in clfs]
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

            real = image.to(device)
            fake = net_G(real)

            loss_g = criterionGAN(iloss, args.ei_array, args.n_classes)
            loss_g_batch += loss_g

            logging.info(
                '[{}]({}/{}) \t {:.4f} '.format(
                    epoch, iteration, len(valid_loader),
                    loss_g
                ) +
                ' '.join(['\t {:.4f} ({:.2f})'.format(
                    l, a) for l, a in zip(iloss, acc)])
            )

        num_samples = len(valid_loader)
        logging.info(
            '[{}](batch) \t {:.4f} '.format(
                epoch,
                loss_g_batch / num_samples
            ) +
            ' '.join(['\t {:.4f} ({:.2f})'.format(
                l/num_samples, a/num_samples) for l, a in zip(loss_m_batch, acc_m_batch)])
        )

        loss_history['valid_epoch'].append(epoch)
        loss_history['valid_G'].append(loss_g_batch/num_samples)
        acc_history['valid_epoch'].append(epoch)
        for idx, (l, a) in enumerate(zip(loss_m_batch, acc_m_batch)):
            loss_history['valid_M_{}'.format(idx)].append(l/num_samples)
            acc_history['valid_M_{}'.format(idx)].append(a/num_samples)

        if epoch in args.save_ckpts:
            model_ckpt = '{}/{}/models/{}_g_{}.stop'.format(
                cfg.ckpt_folder, expt, model_name, epoch)
            logging.info('Model: {}'.format(model_ckpt))
            torch.save(net_G.state_dict(), model_ckpt)

        [sch.step() for sch in sch_clfs]

        if args.net_type in ['linear', 'logistic', 'fcn']:
            continue

        for i in range(image.shape[0]):
            j = np.random.randint(0, image.shape[0])
            sample = image[j]
            label = [str(int(_[j])) for _ in labels]
            ax = plt.subplot(2, 4, i + 1)
            ax.axis('off')
            sample = sample.permute(1, 2, 0)
            plt.imshow(sample.squeeze().numpy())
            plt.savefig(
                '{}/{}/validation/tmp.jpg'.format(cfg.ckpt_folder, expt))
            ax = plt.subplot(2, 4, 5+i)
            ax.axis('off')
            ax.set_title(" ".join(label))
            sample_G = net_G(sample.clone().permute(
                2, 0, 1).unsqueeze_(0).to(device))
            sample_G = sample_G.cpu().detach().squeeze()
            if sample_G.shape[0] == 3:
                sample_G = sample_G.permute(1, 2, 0)
            plt.imshow(sample_G.numpy())

            if i == 3:
                validation_plt = '{}/{}/validation/{}_{}.jpg'.format(
                    cfg.ckpt_folder, expt, model_name, epoch)
                print('Saving: {}'.format(validation_plt))
                plt.tight_layout()
                plt.savefig(validation_plt)
                break

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

    for idx, clf in enumerate(clfs):
        model_ckpt = '{}/{}/models/{}_clf_{}.stop'.format(
            cfg.ckpt_folder, expt, model_name, idx)
        logging.info('Model: {}'.format(model_ckpt))
        torch.save(clf.state_dict(), model_ckpt)


if __name__ == '__main__':
    setup_graceful_exit()
    args = parse()
    model = 'adv_train_eigan_{}_{}'.format(
        args.net_type, args.ckpt_g.split('/')[-1])
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
