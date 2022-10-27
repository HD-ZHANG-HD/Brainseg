import argparse
from cProfile import label
import logging
import os
import copy
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
#from torchvision import transforms
from data_reader import H5DataLoader
import torch.nn.functional as F
from losses import BCE, GCE, SCE, BCE_Weighted
import pdb

from styleaug import StyleAugmentor
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
from augmentation import AddGaussianNoise


def trainer_synapse(args, model, snapshot_path, p=1/3):
    #from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    db_train = H5DataLoader(args.root_path)
    #HDF5Dataset('C:/ml/data', recursive=True, load_data=False, data_cache_size=4, transform=None)
    # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                           transform=transforms.Compose(
    #                               [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train.images)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
    trainloader = db_train
    # worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    if args.loss == 'SCE':
        ce_loss = SCE(args.num_classes, args.device, beta=args.beta)
    elif args.loss == 'GCE':
        ce_loss = GCE(args.num_classes, args.device, q=args.beta)
    elif args.loss == 'BCE':
        ce_loss = BCE(args.num_classes, args.device, beta=args.beta)
    else:
        ce_loss = nn.CrossEntropyLoss()

    criterion_u = nn.CrossEntropyLoss(reduction='none')
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    # max_epoch = max_iterations // len(trainloader) + 1
    max_iterations = args.max_epochs * len(trainloader.images)/args.batch_size
    logging.info("{} iterations per epoch. {} max iterations ".format(
        len(trainloader.images)/args.batch_size, max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    robust_loss = copy.deepcopy(ce_loss)

    ##
    augmentor_s = StyleAugmentor()
    augmentor_w = AddGaussianNoise()
    ##

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(range(np.uint16(len(trainloader.images)/args.batch_size))):
            image_batch, label_batch = trainloader.next_batch(args.batch_size)
            image_batch = image_batch[:, :, :, None]
            image_batch = torch.from_numpy(
                image_batch).permute([0, 3, 1, 2]).to(torch.float)
            label_batch = torch.from_numpy(label_batch).permute(
                [0, 3, 1, 2]).to(torch.float)
# =================== weak aug ==============================
            image_w = augmentor_w(image_batch)
# ==========================================================

            #label_batch = label_batch[:,:,:,4]
            # 将image——batch分成有label和无label两份（chunk（2））

            #image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            if type(image_w) == np.ndarray:
                image_w = torch.from_numpy(image_w).to(torch.float)
            image_l_w, image_u_w = image_w.chunk(2)
            label = label_batch[:image_l_w.shape[0], :, :, :]
            image_l_w, image_u_w, label = image_l_w.cuda(), image_u_w.cuda(), label.cuda()
            _, image_u = image_batch.chunk(2)
            image_u = image_u.cuda()

# =================== strong aug =============================
            image_u_s = torch.cat(
                (image_u, image_u, image_u), 1)
            image_u_s1 = augmentor_s(image_u_s)[:, 0, :, :].unsqueeze(1)
            image_u_s2 = augmentor_s(image_u_s)[:, 1, :, :].unsqueeze(1)


# =================== predict ================================
            pred_l_w = model(image_l_w)

            pred_u_w, pred_u_w_fp = model(image_u_w, True)
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            pred_u_s1 = model(image_u_s1)
            mask_u_s1 = pred_u_s1.argmax(dim=1)

            pred_u_s2 = model(image_u_s2)
            mask_u_s2 = pred_u_s2.argmax(dim=1)

            # outputs = model(image_batch)
            labs = torch.argmax(label, dim=1, keepdim=False)

# =================== loss ====================================
            class_weights = None
            if args.class_weight is not None:
                labs_onehot = torch.nn.functional.one_hot(
                    torch.squeeze(labs), args.num_classes).float()
                tmp = labs_onehot.permute((0, 2, 3, 1))
                class_weights = 1 - \
                    (torch.sum(tmp.reshape((-1, 9)), axis=0) /
                     (labs.shape[0]*labs.shape[1]*labs.shape[2]))

            if epoch_num < args.warmup:
                ce_loss = nn.CrossEntropyLoss(weight=class_weights)
            else:
                ce_loss = robust_loss
                if args.class_weight is not None:
                    ce_loss = BCE_Weighted(
                        args.num_classes, args.device, beta=args.beta, weights=class_weights)

            loss_ce_l = ce_loss(pred_l_w, labs)
            loss_dice_l = dice_loss(
                pred_l_w, labs, weight=class_weights, softmax=True)
            loss_x = 0.5 * loss_ce_l + 0.5 * loss_dice_l

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_s1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w >= args.conf_thresh))
            loss_u_s1 = torch.sum(loss_u_s1) / \
                torch.sum(conf_u_w >= args.conf_thresh).item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_s2)
            loss_u_s2 = loss_u_s1 * ((conf_u_w >= args.conf_thresh))
            loss_u_s2 = torch.sum(loss_u_s2) / \
                torch.sum(conf_u_w >= args.conf_thresh).item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= args.conf_thresh))
            loss_u_w_fp = torch.sum(loss_u_w_fp) / \
                torch.sum(conf_u_w >= args.conf_thresh).item()

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 *
                    0.25 + loss_u_w_fp * 0.5) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_x', loss_x, iter_num)
            writer.add_scalar('info/loss_u_s1', loss_u_s1, iter_num)
            writer.add_scalar('info/loss_u_s2', loss_u_s2, iter_num)
            writer.add_scalar('info/loss_u_w_fp', loss_u_w_fp, iter_num)

            logging.info('iteration %d : loss : %f, loss_x: %f, loss_u_s1: %f, loss_u_s2: %f, loss_u_w_fp: %f' %
                         (iter_num, loss.item(), loss_x.item(), loss_u_s1.item(), loss_u_s2.item(), loss_u_w_fp.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    pred_l_w, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction1',
                                 outputs[1, ...] * 30, iter_num)
                labs1 = labs[1, ...].unsqueeze(0) * 30
                writer.add_image('train/GroundTruth1', labs1, iter_num)

        # save_interval = 50  # int(max_epoch/6)
        # save every epoch epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if 1:
            if args.resume > 0:
                epoch_num = epoch_num + args.resume + 1
            save_mode_path = os.path.join(
                snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:

            save_mode_path = os.path.join(
                snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


# def trainer_synapse(args, model, snapshot_path, p=1/3):
#     #from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
#     logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
#                         format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))
#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     batch_size = args.batch_size * args.n_gpu
#     # max_iterations = args.max_iterations

#     db_train = H5DataLoader(args.root_path)
#     #HDF5Dataset('C:/ml/data', recursive=True, load_data=False, data_cache_size=4, transform=None)
#     # db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
#     #                           transform=transforms.Compose(
#     #                               [RandomGenerator(output_size=[args.img_size, args.img_size])]))
#     print("The length of train set is: {}".format(len(db_train.images)))

#     def worker_init_fn(worker_id):
#         random.seed(args.seed + worker_id)

#     # DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
#     trainloader = db_train
#     # worker_init_fn=worker_init_fn)
#     if args.n_gpu > 1:
#         model = nn.DataParallel(model)
#     model.train()

#     if args.loss == 'SCE':
#         ce_loss = SCE(args.num_classes, args.device, beta=args.beta)
#     elif args.loss == 'GCE':
#         ce_loss = GCE(args.num_classes, args.device, q=args.beta)
#     elif args.loss == 'BCE':
#         ce_loss = BCE(args.num_classes, args.device, beta=args.beta)
#     else:
#         ce_loss = nn.CrossEntropyLoss()

#     dice_loss = DiceLoss(num_classes)
#     optimizer = optim.SGD(model.parameters(), lr=base_lr,
#                           momentum=0.9, weight_decay=0.0001)
#     writer = SummaryWriter(snapshot_path + '/log')
#     iter_num = 0
#     max_epoch = args.max_epochs
#     # max_epoch = max_iterations // len(trainloader) + 1
#     max_iterations = args.max_epochs * len(trainloader.images)/args.batch_size
#     logging.info("{} iterations per epoch. {} max iterations ".format(
#         len(trainloader.images)/args.batch_size, max_iterations))
#     best_performance = 0.0
#     iterator = tqdm(range(max_epoch), ncols=70)
#     robust_loss = copy.deepcopy(ce_loss)

#     ##
#     augmentor_s = StyleAugmentor()
#     augmentor_w = AddGaussianNoise()
#     ##

#     for epoch_num in iterator:
#         for i_batch, sampled_batch in enumerate(range(np.uint16(len(trainloader.images)/args.batch_size))):
#             image_batch, label_batch = trainloader.next_batch(args.batch_size)
#             image_batch = image_batch[:, :, :, None]
# # =================== weak aug ==============================
#             image_batch = augmentor_w(image_batch)
# # ==========================================================
#             image_batch = torch.from_numpy(
#                 image_batch).permute([0, 3, 1, 2]).to(torch.float)
#             label_batch = torch.from_numpy(label_batch).permute(
#                 [0, 3, 1, 2]).to(torch.float)
#             #label_batch = label_batch[:,:,:,4]
#             # 将image——batch分成有label和无label两份（chunk（2））

#             #image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

# # =================== strong aug =============================
#             if np.random.uniform(0, 1) < p:
#                 image_batch = torch.cat(
#                     (image_batch, image_batch, image_batch), 1)
#                 image_batch = augmentor_s(image_batch)[:, 1, :, :].unsqueeze(1)

# # ============================================================
#             outputs = model(image_batch)
#             labs = torch.argmax(label_batch, dim=1, keepdim=False)

#             class_weights = None
#             if args.class_weight is not None:
#                 labs_onehot = torch.nn.functional.one_hot(
#                     torch.squeeze(labs), args.num_classes).float()
#                 tmp = labs_onehot.permute((0, 2, 3, 1))
#                 class_weights = 1 - \
#                     (torch.sum(tmp.reshape((-1, 9)), axis=0) /
#                      (labs.shape[0]*labs.shape[1]*labs.shape[2]))

#             if epoch_num < args.warmup:
#                 ce_loss = nn.CrossEntropyLoss(weight=class_weights)
#             else:
#                 ce_loss = robust_loss
#                 if args.class_weight is not None:
#                     ce_loss = BCE_Weighted(
#                         args.num_classes, args.device, beta=args.beta, weights=class_weights)

#             loss_ce = ce_loss(outputs, labs)
#             loss_dice = dice_loss(
#                 outputs, labs, weight=class_weights, softmax=True)
#             loss = 0.5 * loss_ce + 0.5 * loss_dice
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_

#             iter_num = iter_num + 1
#             writer.add_scalar('info/lr', lr_, iter_num)
#             writer.add_scalar('info/total_loss', loss, iter_num)
#             writer.add_scalar('info/loss_ce', loss_ce, iter_num)

#             logging.info('iteration %d : loss : %f, loss_ce: %f' %
#                          (iter_num, loss.item(), loss_ce.item()))

#             if iter_num % 20 == 0:
#                 image = image_batch[1, 0:1, :, :]
#                 image = (image - image.min()) / (image.max() - image.min())
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(
#                     outputs, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/Prediction1',
#                                  outputs[1, ...] * 30, iter_num)
#                 labs1 = labs[1, ...].unsqueeze(0) * 30
#                 writer.add_image('train/GroundTruth1', labs1, iter_num)

#         # save_interval = 50  # int(max_epoch/6)
#         # save every epoch epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
#         if 1:
#             if args.resume > 0:
#                 epoch_num = epoch_num + args.resume + 1
#             save_mode_path = os.path.join(
#                 snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))

#         if epoch_num >= max_epoch - 1:

#             save_mode_path = os.path.join(
#                 snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
#             torch.save(model.state_dict(), save_mode_path)
#             logging.info("save model to {}".format(save_mode_path))
#             iterator.close()
#             break

#     writer.close()
#     return "Training Finished!"
