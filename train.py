import argparse
import os
from string import whitespace

import paddle
import paddle.nn as nn
from paddle.optimizer import SGD, Adam
from tqdm import tqdm
from visualdl import LogWriter

from dataloader import make_data_loader
from model.linkcrack import *
from utils.lr_scheduler import LR_Scheduler as Sche


def build_parser():
    parser = argparse.ArgumentParser(description="LinkCrack Training")

    parser.add_argument('--dataset', type=str, default='TunnelCrack',
                        choices=['TunnelCrack'],
                        help='dataset name (default: TunnelCrack)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--train-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--val-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--use_adam', type=str, default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--cuda', action='store_true',
                        default=False, help='Use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--pretrained_model', type=str,
                        default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='LinkCrack',
                        help='set the checkpoint name')
    parser.add_argument('--checkpath', type=str, default='checkpoints',
                        help='save checkpoints path')
    parser.add_argument('--max_save', type=int, default=20,
                        help='maximum number of checkpoints to be saved')
    # visdom
    parser.add_argument('--port', type=int, default=8097,
                        help='visdom port')
    parser.add_argument('--vis_train_loss_every', type=int, default=100,
                        help='the logger interval for loss')
    parser.add_argument('--vis_train_acc_every', type=int, default=100,
                        help='the logger interval for acc')
    parser.add_argument('--vis_train_img_every', type=int, default=200,
                        help='image interval')

    # eval
    parser.add_argument('--val_every', type=int, default=600,
                        help='evaluuation interval')

    # loss
    parser.add_argument('--loss_weight', type=int, default=10,
                        help='the weight of loss')
    parser.add_argument('--pos_pixel_weight', type=int, default=1,
                        help='the weight of positive pixel loss')
    parser.add_argument('--pos_link_weight', type=int, default=10,
                        help='the weight of positive link pixel loss')
    parser.add_argument('--acc_sigmoid_th', type=float, default=0.5,
                        help='the threshold of pixel confidence in loss')

    args = parser.parse_args()
    return args


def main():
    args = build_parser()
    place = paddle.CUDAPlace(0) if args.cuda else paddle.CPUPlace()
    paddle.disable_static(place)

    if not os.path.exists(args.checkpath):
        os.mkdir(args.checkpath)

    args.saver_path = os.path.join(args.checkpath, args.checkname)
    writer = LogWriter("./log/Train")
    paddle.seed(args.seed)
    model = LinkCrack()

    if args.pretrained_model:
        model.load_state_dict(paddle.load(args.pretrained_model))
        print('load pretrained model from {}'.format(args.pretrained_model))
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(
        args, **kwargs)
    lr_scheduler = Sche(args.lr_scheduler, args.lr,
                        args.epochs, len(train_loader))
    if args.use_adam:
        optimizer = Adam(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay)
    else:
        optimizer = SGD(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay)

        mask_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=paddle.to_tensor([
                                         args.pos_pixel_weight], dtype=paddle.float32))
        connected_loss = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=paddle.to_tensor([
                                              args.pos_link_weight], dtype=paddle.float32))

    def acc(pred, mask, connected):
        pred = paddle.nn.functional.sigmoid(pred)
        pred = paddle.round(pred)
        pred_mask = pred[:, 0]
        pred_connected = pred[:, 1:]
        mask = mask[:, 0]
        mask_acc = paddle.equal(pred_mask, mask)
        mask_acc.sum() / mask.numel()
        mask_pos_acc = pred_mask[mask > 0].equal(mask[mask > 0]).sum() / mask[
            mask > 0].numel()
        mask_neg_acc = pred_mask[mask < 1].equal(mask[mask < 1]).sum() / mask[
            mask < 1].numel()
        connected_acc = pred_connected.equal(
            connected).sum() / connected.numel()
        connected_pos_acc = pred_connected[connected > 0].equal(
            connected[connected > 0]).sum() / connected[connected > 0].numel()
        connected_neg_acc = pred_connected[connected < 1].equal(
            connected[connected < 1]).sum() / connected[connected < 1].numel()

        log_acc = {
            'mask_acc': mask_acc,
            'mask_pos_acc': mask_pos_acc,
            'mask_neg_acc': mask_neg_acc,
            'connected_acc': connected_acc,
            'connected_pos_acc': connected_pos_acc,
            'connected_neg_acc': connected_neg_acc
        }
        return log_acc

    total_iter = 0
    save_pos_acc = 0
    # ---------train----------
    for epoch in range(args.epochs):
        model.train()
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        bar.set_description('Epoch %d --- Training --- :' % epoch)
        for idx, sample in bar:
            img = sample['image']
            lab = sample['label']
            lab = paddle.to_tensor(lab, dtype=paddle.float32)
            mask, connected = lab[0], lab[1]
            optimizer.clear_grad()
            pred_mask, pred_connected = model(img)
            mask_loss = mask_loss(paddle.reshape(
                pred_mask, [-1, 1]), paddle.reshape(mask, [-1, 1])) / args.train_batch_size
            connected_loss = connected_loss(paddle.reshape(
                pred_connected, [-1, 1]), paddle.reshape(connected, [-1, 1])) / args.train_batch_size

            total_loss = mask_loss + args.loss_weight * connected_loss
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            pred = paddle.concat([pred_mask, pred_connected], axis=1)
            total_iter += 1
            if total_iter % args.vis_train_loss_every == 0:
                writer.add_scalar(
                    tag='mask_loss', step=total_iter, value=mask_loss.numpy())
                writer.add_scalar(tag='connected_loss',
                                  step=total_iter, value=connected_loss.numpy())
                writer.add_scalar(tag='total_loss',
                                  step=total_iter, value=total_loss.numpy())

            if total_iter % args.vis_train_acc_every == 0:
                log_acc = acc(pred, mask, connected)
                writer.add_scalar(
                    tag='mask_acc', step=total_iter, value=log_acc['mask_acc'])
                writer.add_scalar(
                    tag='mask_pos_acc', step=total_iter, value=log_acc['mask_pos_acc'])
                writer.add_scalar(
                    tag='mask_neg_acc', step=total_iter, value=log_acc['mask_neg_acc'])
                writer.add_scalar(
                    tag='connected_acc', step=total_iter, value=log_acc['connected_acc'])
                writer.add_scalar(
                    tag='connected_pos_acc', step=total_iter, value=log_acc['connected_pos_acc'])
                writer.add_scalar(
                    tag='connected_neg_acc', step=total_iter, value=log_acc['connected_neg_acc'])

            if idx % args.vis_train_img_every == 0:
                writer.add_image(tag='train_image',
                                 img=tensor2image(img[0]), step=total_iter)
                writer.add_image(tag='train_pred', img=tensor2image(
                    pred[0, 0]), step=total_iter)
                writer.add_image(tag='train_mask', img=tensor2image(
                    mask[0]), step=total_iter)
                writer.add_image(tag='train_connected_0', img=tensor2image(
                    connected[0][0]), step=total_iter)
                writer.add_image(tag='train_connected_1', img=tensor2image(
                    connected[0][1]), step=total_iter)
                writer.add_image(tag='train_connected_2', img=tensor2image(
                    connected[0][2]), step=total_iter)
                writer.add_image(tag='train_connected_3', img=tensor2image(
                    connected[0][3]), step=total_iter)
                writer.add_image(tag='train_connected_4', img=tensor2image(
                    connected[0][4]), step=total_iter)
                writer.add_image(tag='train_connected_5', img=tensor2image(
                    connected[0][5]), step=total_iter)
                writer.add_image(tag='train_connected_6', img=tensor2image(
                    connected[0][6]), step=total_iter)
                writer.add_image(tag='train_connected_7', img=tensor2image(
                    connected[0][7]), step=total_iter)
                writer.add_image(tag='train_link_0', img=tensor2image(
                    paddle.nn.functional.sigmoid(pred[0, 1])), step=total_iter)
                writer.add_image(tag='train_link_1', img=tensor2image(
                    paddle.nn.functional.sigmoid(pred[0, 2])), step=total_iter)
                writer.add_image(tag='train_link_2', img=tensor2image(
                    paddle.nn.functional.sigmoid(pred[0, 3])), step=total_iter)
                writer.add_image(tag='train_link_3', img=tensor2image(
                    paddle.nn.functional.sigmoid(pred[0, 4])), step=total_iter)
                writer.add_image(tag='train_link_4', img=tensor2image(
                    paddle.nn.functional.sigmoid(pred[0, 5])), step=total_iter)
                writer.add_image(tag='train_link_5', img=tensor2image(
                    paddle.nn.functional.sigmoid(pred[0, 6])), step=total_iter)
                writer.add_image(tag='train_link_6', img=tensor2image(
                    paddle.nn.functional.sigmoid(pred[0, 7])), step=total_iter)
                writer.add_image(tag='train_link_7', img=tensor2image(
                    paddle.nn.functional.sigmoid(pred[0, 8])), step=total_iter)

            if total_iter % args.val_every == 0:
                print("start val")
                model.eval()
                bar.set_description('Epoch %d --- Evaluation --- :' % epoch)
                val_loss = {
                    'mask_loss': 0,
                    'connect_loss': 0,
                    'total_loss': 0
                }
                val_acc = {
                    'mask_acc': 0,
                    'mask_pos_acc': 0,
                    'mask_neg_acc': 0,
                    'connected_acc': 0,
                    'connected_pos_acc': 0,
                    'connected_neg_acc': 0
                }
                with paddle.no_grad():
                    for idx, sample in enumerate(val_loader, start=1):
                        img = sample['image']
                        lab = sample['label']
                        img = paddle.to_tensor(img, dtype=paddle.float32)

                        mask = lab[0]
                        connected = lab[1]

                        pred_mask, pred_connected = model(img)
                        pred = paddle.concat(
                            [pred_mask, pred_connected], axis=1)
                        log_acc = acc(pred, mask, connected)
                        mask_loss = mask_loss(paddle.reshape(
                            pred_mask, [-1, 1]), paddle.reshape(mask, [-1, 1])) / args.train_batch_size
                        connected_loss = connected_loss(paddle.reshape(
                            pred_connected, [-1, 1]), paddle.reshape(connected, [-1, 1])) / args.train_batch_size

                        total_loss = mask_loss + args.loss_weight * connected_loss

                        val_loss['mask_loss'] += mask_loss.numpy()
                        val_loss['connect_loss'] += connected_loss.numpy()
                        val_loss['total_loss'] += total_loss.numpy()
                        val_acc['mask_acc'] += log_acc['mask_acc']
                        val_acc['connected_acc'] += log_acc['connected_acc']
                        val_acc['mask_pos_acc'] += log_acc['mask_pos_acc']
                        val_acc['connected_pos_acc'] += log_acc['connected_pos_acc']
                        val_acc['mask_neg_acc'] += log_acc['mask_neg_acc']
                        val_acc['connected_neg_acc'] += log_acc['connected_neg_acc']
                    else:
                        writer.add_image(tag='val_image', img=tensor2image(
                            img[0]), step=total_iter)
                        writer.add_image(tag='val_mask', img=tensor2image(
                            mask[0]), step=total_iter)
                        writer.add_image(tag='val_pred_mask', img=tensor2image(
                            pred_mask[0]), step=total_iter)

                        writer.add_scalar(
                            tag='val_mask_loss', step=total_iter, value=val_loss['mask_loss'] / idx)
                        writer.add_scalar(
                            tag='val_connect_loss', step=total_iter, value=val_loss['connect_loss'] / idx)
                        writer.add_scalar(
                            tag='val_total_loss', step=total_iter, value=val_loss['total_loss'] / idx)

                        writer.add_scalar(
                            tag='val_mask_acc', step=total_iter, value=val_acc['mask_acc'] / idx)
                        writer.add_scalar(
                            tag='val_connected_acc', step=total_iter, value=val_acc['connected_acc'] / idx)
                        writer.add_scalar(
                            tag='val_mask_pos_acc', step=total_iter, value=val_acc['mask_pos_acc'] / idx)
                        writer.add_scalar(
                            tag='val_connected_pos_acc', step=total_iter, value=val_acc['connected_pos_acc'] / idx)
                        writer.add_scalar(
                            tag='val_mask_neg_acc', step=total_iter, value=val_acc['mask_neg_acc'] / idx)
                        writer.add_scalar(
                            tag='val_connected_neg_acc', step=total_iter, value=val_acc['connected_neg_acc'] / idx)

                bar.set_description('Epoch %d --- Training --- :' % epoch)

                # -----save model-----
                if (val_acc['mask_pos_acc'] / idx) > save_pos_acc:
                    save_pos_acc = val_acc['mask_pos_acc'] / idx
                    save_path = os.path.join(
                        args.saver_path, 'model_pos_acc_%.4f_weight_%f.pdparams' % (save_pos_acc, args.loss_weight))
                    paddle.save(model.state_dict(), save_path)
                    print("save checkpoint to ", save_path)
                model.train()


def tensor2image(x):
    x *= 255
    x = paddle.clip(0, 255)
    x = paddle.transpose(x, [1, 2, 0])
    return x.numpy().astype(np.uint8)


if __name__ == '__main__':
    main()
