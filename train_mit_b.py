#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 19:59:31 2019

@author: hi
"""

# System libs
import os
import time
import timeit
# import math
import random
import argparse
# import cv2
import numpy as np
from packaging import version

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Numerical libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils import data, model_zoo


from scipy.io import loadmat
from scipy.misc import imresize, imsave

# Our libs
from datasets.mit_sceneparsing_benchmark_dataset import MITSceneParsingBenchmarkDataset
from utils.utils import AverageMeter, colorEncode, accuracy , intersectionAndUnion, pixel_acc, as_numpy
from utils.loss import CrossEntropyLoss
from utils.parallel import DataParallelModel, DataParallelCriterion
from models import get_model
# from lib.utils.th import as_numpy

start_time = timeit.default_timer()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


RESTORE_FROM = "./models/pretrained/resnet50-imagenet.pth"
#RESTORE_FROM = "./models/pretrained/resnet101-imagenet.pth"


def create_optimizers(models, criterion, args):
    optimizer_model = torch.optim.SGD(
            models.parameters(),
            lr = args.learning_rate,
            momentum = args.momentum,
            weight_decay = args.weight_decay)
    optimizer_model.zero_grad()
    
    return optimizer_model

def adjust_learning_rate(optimizers, current_iter, args):
    scale_running_lr = ((1. - float(current_iter) / args.max_iters)**(args.power))
    args.running_lr = args.learning_rate * scale_running_lr
    
    for param_group in optimizers.param_groups:
        param_group['lr'] = args.running_lr
        
def checkpoint(models, history, epoch_num, args):
    suffix_latest = 'epoch{}.pth'.format(epoch_num)
    suffix_best = 'best.pth'
    
    if args.num_gpus > 1:
        dict_models = models.module.state_dict()
    else:
        dict_models = models.state_dict()
        
    if epoch_num % 10 == 0 or epoch_num == args.num_epoches: 
        print('Saving checkpoints at {} epochs ...'.format(epoch_num))
        torch.save(history, '{}/history_{}'.format(args.checkpoints_dir, suffix_latest))
        torch.save(dict_models, '{}/{}_{}'.format(args.checkpoints_dir, args.model, suffix_latest))
    
    current_loss = history['val']['loss'][-1]
    if current_loss < args.best_loss:
        args.best_loss = current_loss
        print('Saving checkpionts bset ...')
        torch.save(dict_models, '{}/{}_{}'.format(args.checkpoints_dir, args.model, suffix_best))

def visualize_result(batch_data, pred, args):
    colors = loadmat('datasets/mit_list/color150.mat')['colors']
    (image, seg, info) = batch_data
    
    for j in range(len(info)):
        # recover image
        img = image[j].clone()
        for t, m, s in zip(img,
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        img = img[:, :, ::-1]
        
        # segmentation
        seg_ = seg[j].numpy()
        seg_color = colorEncode(seg_, colors)
        
        # preddiction
        pred_ = np.argmax(pred.data.cpu()[j].numpy(), axis=0)
        pred_color = colorEncode(pred_, colors)
        
        # aggregate images and save
        im_vis = np.concatenate((img, seg_color, pred_color), axis=1).astype(np.uint8)
        
        img_name = info[j].split('/')[-1]
        imsave(os.path.join(args.result, img_name.replace('.jpg', '.png')), im_vis)
        #cv2.imwrite(os.path.join(args.result, img_name.replace('.jpg', '.png')), im_vis)
        
def visualize_result_simple(batch_data, pred, args):
    colors = loadmat('datasets/mit_list/color150.mat')['colors']
    (image, seg, info) = batch_data
    
    for j in range(len(info)):
        # recover image
        img = image[j].clone()
        for t, m, s in zip(img,
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        img = img[:, :, ::-1]
        
        # segmentation
        seg_ = seg[j].numpy()
        seg_color = colorEncode(seg_, colors)
        
        # preddiction
        # pred_ = np.argmax(pred.data.cpu()[j].numpy(), axis=0)
        pred_ = np.argmax(pred.data.numpy(), axis=0)
        pred_color = colorEncode(pred_, colors)
        
        # aggregate images and save
        im_vis = np.concatenate((img, seg_color, pred_color), axis=1).astype(np.uint8)
        
        img_name = info[j].split('/')[-1]
        imsave(os.path.join(args.result, img_name.replace('.jpg', '.png')), im_vis)
        #cv2.imwrite(os.path.join(args.result, img_name.replace('.jpg', '.png')), im_vis)
    
    
# train one epoch
def train(models, train_loader, interp, optimizers, criterion, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # loss_value = 0
    
    # Switch to train mode
    models.train()

    # main loop
    tic = time.time()    
    for i_iter, batch_data in enumerate(train_loader):
        cur_iter = i_iter + (epoch - 1) * args.epoch_iters
        # measure data loading time
        torch.cuda.synchronize()
        data_time.update(time.time() - tic)
        
        # optimizers.zero_grad()
        # cur_iter = i_iter + (epoch - 1) * args.epoch_iters
        # adjust_learning_rate(optimizers, cur_iter, args)
        
        # forward pass
        images, labels, _ = batch_data
        # print(images.type())
        
        # feed input data
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images = images.to(device)
        labels = labels.to(device)
        #print(labels.shape)
        #print(labels.size())
        #print(labels)
        
        optimizers.zero_grad()
        adjust_learning_rate(optimizers, cur_iter, args)
        
        pred_seg = models(images)
        #print(pred_seg)
        
        pred_seg = interp(pred_seg)
        # pred_seg = F.softmax(pred_seg)
        
        loss = criterion(pred_seg, labels)
        # loss_value += loss.item()
        #print(loss)
        # acc = pixel_acc(pred_seg, labels)
        # acc, _ = accuracy(pred_seg, labels)
        
        # loss = loss.mean()
        # acc = acc.mean()
        
        # Backward / compute gradient and do SGD step
        # optimizers.zero_grad()
        loss.backward()
        # optimizers.step()
        #ave_total_loss.update(loss.data.item())
        #ave_acc.update(acc.data.item() * 100)
        
        # loss_value += loss.data.cpu().numpy().item()
        # loss_value += loss.data.item()
        # loss_value += loss.item()
        # loss_value += loss.data.cpu().numpy()[0]
        # loss_value += loss.cpu().numpy()[0]
        
        optimizers.step()
        
        #loss_value += loss.item()
        # loss_value += loss.data.cpu().numpy().item()
        # loss_value = loss.data.item()
        
        # Measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        
        #loss_value += loss.data.cpu().numpy().item()
        
        # Update average loss and acc
        # acc = pixel_acc(pred_seg, labels)
        # ave_total_loss.update(loss.data.item())
        # ave_acc.update(acc.data.item() * 100)
        
        
        if  i_iter % args.display_iter == 0:
            acc = pixel_acc(pred_seg, labels)
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'LR: {:.6f}  '
                  'Accurary: {:4.2f}, Loss: {:.6f}  '
                  .format(epoch, i_iter, args.epoch_iters, 
                          batch_time.average(), data_time.average(),
                          args.running_lr, 
                          acc.data.item() * 100, loss.data.item() ))
        
            fractional_epoch = epoch - 1 + 1. * i_iter / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())
        
        # Adjust learning rate
        # cur_iter = i_iter + (epoch - 1) * args.epoch_iters
        # adjust_learning_rate(optimizers, cur_iter, args)
        
def evaluate(models, val_loader, interp, criterion, history, epoch, args):
    print('***Evaluating at {} epoch ...'.format(epoch))
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    models.eval()
    
    for i, batch_data in enumerate(val_loader):
        torch.cuda.synchronize()
        # forward pass
        images, labels, _ = batch_data
        
        images = images.to(device)
        labels = labels.to(device)
        
        pred_seg = models(images)
        pred_seg = interp(pred_seg)
        # pred_seg = F.softmax(pred_seg)
        
        loss = criterion(pred_seg, labels)
        loss_meter.update(loss.data.item())
        print('[Eval] iter {}, loss: {}'.format(i, loss.data.item()))
        
        #acc = pixel_acc(pred_seg, labels)
        #acc_meter.update(acc.data.item())
        
        labels = as_numpy(labels)
        _, pred = torch.max(pred_seg, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())
        acc, pix = accuracy(pred, labels)
        acc_meter.update(acc, pix)        
        
        if args.visualize:
            visualize_result(batch_data, pred_seg, args)
    
    history['val']['epoch'].append(epoch)
    history['val']['loss'].append(loss_meter.average())
    history['val']['acc'].append(acc_meter.average())
    print('[Eval Summary] Epoch: {}, Loss: {}, Accurarcy: {:4.2f}%'
          .format(epoch, loss_meter.average(), acc_meter.average()*100))

    # Plot figure
    if epoch > 0:
        print('Plotting loss figure...')
        fig = plt.figure()
        plt.plot(np.asarray(history['train']['epoch']),
                 np.log(np.asarray(history['train']['loss'])),
                 color='b', label='training')
        plt.plot(np.asarray(history['val']['epoch']),
                 np.log(np.asarray(history['val']['loss'])),
                 color='c', label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Log(loss)')
        fig.savefig('{}/loss.png'.format(args.checkpoints_dir), dpi=225)
        plt.close('all')

        fig = plt.figure()
        plt.plot(history['train']['epoch'], history['train']['acc'],
                 color='b', label='training')
        plt.plot(history['val']['epoch'], history['val']['acc'],
                 color='c', label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        fig.savefig('{}/accuracy.png'.format(args.checkpoints_dir), dpi=225)
        plt.close('all')



def main(args):
    # cudnn.enabled = True
    cudnn.benchmark = True
    
    # Setup Models
    model = get_model(name=args.model, num_classes = args.num_classes)
    
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
        
    # Only copy the params that exist in current model
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        # print(name)
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
            # print('copy {}'.format(name))
    model.load_state_dict(new_params)
    
    # model.train()
    
    # load model into GPU
    print('GPU numbers: [ {} ] !'.format(torch.cuda.device_count()))  # GPU number
    if torch.cuda.device_count() > 1:
        #model = torch.nn.DataParallel(model, device_ids=range(args.num_gpus))
        # model = DataParallelModel(model, device_ids=range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    #model.to(device)
    
    # cudnn.benchmark = True # acceleration
    
    # Dataset and Loader
    train_dataset = MITSceneParsingBenchmarkDataset(root=args.root_dataset, split="training", img_size=args.imgSize)
    train_loader = data.DataLoader(train_dataset,
                                  batch_size = args.batch_size,
                                  shuffle = True,
                                  num_workers = int(args.workers),
                                  pin_memory = True,
                                  drop_last = True)
    
    val_dataset = MITSceneParsingBenchmarkDataset(root=args.root_dataset, split="validation", img_size=args.imgSize, max_sample=args.num_val)
    val_loader = data.DataLoader(val_dataset,
                                  batch_size = 8, # args.batch_size 8
                                  shuffle = False,
                                  num_workers = 8,
                                  pin_memory = True,
                                  drop_last = True)
    
    print('1 Epoch = [ {} ] iters'.format(args.epoch_iters))
    
    # Create loader iterator
    # iterator_train = enumerate(train_loader)
    #iterator_train = iter(train_loader)
    
    
    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # crit = CrossEntropyLoss(ignore_index=-1)
    
    
    # Set up optimizers
    optimizer = create_optimizers(model, crit, args)
    
    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = torch.nn.Upsample(size=(args.imgSize[0], args.imgSize[1]), mode='bilinear', align_corners=True)
    else:
        interp = torch.nn.Upsample(size=(args.imgSize[0], args.imgSize[1]), mode='bilinear')
        # interp = F.interpolate(size=(args.imgSize[0], args.imgSize[1]), mode='bilinear') # umsample
    
    # Main loop
    history = {split: {'epoch': [], 'loss': [], 'acc': []} for split in ('train', 'val')}
    
    # initial eval
    evaluate(model, val_loader, interp, crit, history, 0, args)
    for epoch in range(args.start_epoch, args.num_epoches + 1):
        train(model, train_loader, interp, optimizer, crit, history, epoch, args)
        
        if epoch % args.eval_epoch == 0:
            evaluate(model, val_loader, interp, crit, history, epoch, args)
            
        # checkpointing
        checkpoint(model, history, epoch, args)
            
    end_time = timeit.default_timer()
    print('Running time(h): [{0:.4f}]h'.format((end_time - start_time) / 3600))
    print('Training Done! ***')


if __name__ == '__main__':
    """
    return: A list of parsed arguments
    """
    parser = argparse.ArgumentParser(description="AttentionSegmentationModel")
    # Model related arguments
    parser.add_argument("--model", type=str, default='DeepLab50', 
                        help="Availabel options: ATDNext50, PSPNet50, DeepLab50,DeepLabv3_50...")
    parser.add_argument("--id", type=str, default='ADE2016', 
                        help="A name for identifying the model.")
    
    # optimization related arguments
    parser.add_argument('--num_gpus', type=int, default=2, 
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu', type=int, default=8, 
                        help='input batch size')
    parser.add_argument('--num_epoches', type=int, default=100, # 20 50 60 70 80 100
                        help='epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=1, # 1
                        help='epoch to start training. useful if continue from a checkpoint')
    """
    parser.add_argument('--epoch_iters', type=int, default=5000, 
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument("--dilate_scale", type=int, default=8, 
                        help="Availabel options: 8, 16, 1.")
    """
    
    parser.add_argument("--learning-rate", type=float, default=3.5e-3, 
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.9, 
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, 
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--power", type=float, default=0.9, 
                        help="Decay parameter to compute the learning the learning rate.")
    
    parser.add_argument("--best-loss", type=float, default=0.99,
                        help="Initialize with a big number.")
    
    # Data related arguments 
    parser.add_argument('--num_classes', type=int, default=150, 
                        help='number of classes')
    parser.add_argument('--imgSize', type=tuple, default=(352, 352),  # (512, 683) (224, 224)
                        help='input image size')
    parser.add_argument('--workers', type=int, default=16, 
                        help='number of data loading workers')
    parser.add_argument('--num_val', type=int, default=520,   # 128
                        help='number of images to evalutate')
    
    # path related arguments
    parser.add_argument("--root-dataset", type=str, default="./data/ADEChallengeData2016/",
                        help="With all your images in an images/directory off the root of your site")
    parser.add_argument("--restore-from", type=str, default= RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--checkpoints_dir", type=str, default='./checkpoints/',
                        help="Where to save checkpoints of the model.")
    parser.add_argument("--result", type=str, default="./result/train",
                        help="Folder to output visualization results.")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=304, # 304
                        help="Manual seed")
    parser.add_argument("--eval-epoch", type=int, default=1,
                        help="Frequency to evaluate")
    parser.add_argument("--visualize", type=int, default=1,
                        help="Output visualation: 0 or 1")
    parser.add_argument("--display-iter", type=int, default=20,
                        help="Frequency to display")    
    
    args = parser.parse_args()
    
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    
    # iterations of each epoch (irrelevant to batch size)
    args.samples_size = 20210 # ADE20K 20210 images
    args.epoch_iters = int(args.samples_size / args.batch_size)
    
    args.max_iters = args.epoch_iters * args.num_epoches
    args.running_lr = args.learning_rate
    
    args.id += '-' + str(args.model)
    args.id += '-ngpus' + str(args.num_gpus)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-h' + str(args.imgSize[0])
    args.id += '-w' + str(args.imgSize[1])
    args.id += '-lr' + str(args.learning_rate)
    args.id += '-epoch' + str(args.num_epoches)
    args.id += '-BN_b'        # BN or SyncBN
    
    print('Model ID: {}'.format(args.id))
    
    args.checkpoints_dir = os.path.join(args.checkpoints_dir, args.id)
    args.result = os.path.join(args.result, args.model)
    
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if not os.path.exists(args.result):
        os.makedirs(args.result)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)
    
    #model = get_model(name=args.model, num_classes = args.num_classes)
    #x = torch.randn([2, 3, 512, 512])
    #model(x)
    #print(model)
    #pass
    