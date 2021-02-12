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
    num_clfs = len(args.n_classes)
    net_G = define_G(cfg.num_channels[expt],
                     cfg.num_channels[expt],
                     64, gpu_id=device)
    net_D = define_D(cfg.num_channels[expt],
                     64, 'basic', gpu_id=device)
    net_G = nn.DataParallel(net_G, device_ids=args.gpu_id)
    net_D = nn.DataParallel(net_D, device_ids=args.gpu_id)

    if args.load_w:
        print("Loading weights...\n{}\n{}".format(args.ckpt_g, args.ckpt_d))
        net_G.load_state_dict(torch.load(args.ckpt_g))
        net_D.load_state_dict(torch.load(args.ckpt_d))
    elif args.init_w:
        print("Init weights...")
        net_G.apply(weights_init)
        net_D.apply(weights_init)

    if args.optimizer == 'sgd':
        optim = torch.optim.SGD
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam
    opt_G = torch.optim.Adam(net_G.parameters(), lr=args.lr_g,
                             weight_decay=args.weight_decays[0])
    opt_D = torch.optim.Adam(net_D.parameters(), lr=args.lr_d,
                             weight_decay=args.weight_decays[1])

    criterionGAN = GANLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)

    train_loader = get_loader(expt, args.batch_size, True,
                              img_size=args.img_size, subset=args.subset)
    valid_loader = get_loader(expt, args.test_batch_size,
                              False, img_size=args.img_size, subset=args.subset)

    loss_history = defaultdict(list)
    acc_history = defaultdict(list)
    for epoch in range(args.n_epochs):
        logging.info("Train Epoch \t Loss_D \t Loss_G")

        for iteration, (image, labels) in enumerate(train_loader, 1):
            real = image.to(device)
            fake = net_G(real)
            opt_D.zero_grad()
            pred_fake = net_D.forward(fake.detach())
            loss_d_fake = criterionGAN(pred_fake, False)
            pred_real = net_D.forward(real)
            loss_d_real = criterionGAN(pred_real, True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            opt_D.step()

            opt_G.zero_grad()
            pred_fake = net_D.forward(fake)
            loss_g_gan = criterionGAN(pred_fake, True)
            loss_g_l1 = criterionL1(fake, real) * 10
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            opt_G.step()

            lossd = loss_d.item()
            lossg = loss_g.item()

            logging.info(
                '[{}]({}/{}) \t {:.4f} \t {:.4f} '.format(
                    epoch, iteration, len(train_loader),
                    lossd, lossg
                ) +
                ' '.join(['\t {:.4f} ({:.2f})'.format(
                    l, a) for l, a in zip(iloss, acc)])
            )

        loss_history['train_epoch'].append(epoch)
        loss_history['train_G'].append(lossg)
        loss_history['train_D'].append(lossd)
        acc_history['train_epoch'].append(epoch)

        logging.info("Valid Epoch \t Loss_D \t Loss_G")

        loss_d_batch = 0
        loss_g_batch = 0
        for iteration, (image, labels) in enumerate(valid_loader, 1):
            real = image.to(device)
            fake = net_G(real)

            pred_fake = net_D.forward(fake.detach())
            loss_d_fake = criterionGAN(pred_fake, False)
            pred_real = net_D.forward(real)
            loss_d_real = criterionGAN(pred_real, True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d_batch += loss_d.item()

            pred_fake = net_D.forward(fake)
            loss_g_gan = criterionGAN(pred_fake, True)
            loss_g_l1 = criterionL1(fake, real) * 10
            loss_g = loss_g_gan + loss_g_l1
            loss_g_batch += loss_g.item()

            lossd = loss_d.item()
            lossg = loss_g.item()

        num_samples = len(valid_loader)
        logging.info(
            '[{}](batch) \t {:.4f} \t {:.4f} '.format(
                epoch,
                loss_d_batch / num_samples,
                loss_g_batch / num_samples
            )
        )

        loss_history['valid_epoch'].append(epoch)
        loss_history['valid_G'].append(loss_g_batch/num_samples)
        loss_history['valid_D'].append(loss_d_batch/num_samples)
        acc_history['valid_epoch'].append(epoch)

        for i in range(image.shape[0]):
            j = np.random.randint(0, image.shape[0])
            sample = image[j]
            sample_G = net_G(sample.unsqueeze_(0).to(device))
            ax = plt.subplot(2, 4, i + 1)
            plt.tight_layout()
            ax.axis('off')
            sample = sample.squeeze()
            if sample.shape[0] == 3:
                sample = sample.permute(1, 2, 0)
            plt.imshow(sample.numpy())
            ax = plt.subplot(2, 4, 5+i)
            plt.tight_layout()
            ax.axis('off')
            sample_G = sample_G.cpu().detach().squeeze()
            if sample_G.shape[0] == 3:
                sample_G = sample_G.permute(1, 2, 0)
            plt.imshow(sample_G.numpy())

            if i == 3:
                validation_plt = '{}/{}/validation/{}_{}.jpg'.format(
                    cfg.ckpt_folder, expt, model_name, epoch)
                print('Saving: {}'.format(validation_plt))
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

    model_ckpt = '{}/{}/models/{}_d.stop'.format(
        cfg.ckpt_folder, expt, model_name)
    logging.info('Model: {}'.format(model_ckpt))
    torch.save(net_D.state_dict(), model_ckpt)


if __name__ == '__main__':
    setup_graceful_exit()
    args = parse()
    model = '{}_{}'.format(
        'pretrain_gan', args.net_type)
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
