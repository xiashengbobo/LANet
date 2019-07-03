#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 22:07:14 2019

@author: hi
"""

# System libs
import os
#import sys
import time
import timeit
#import datetime
import argparse
import numpy as np
from packaging import version
#from collections import OrderecdDict

# Numerical libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data, model_zoo
#import torchvision.models as models

from scipy.io import loadmat
from scipy.misc import imsave
from scipy.ndimage import zoom

#from PIL import Image
import matplotlib.pyplot as plt

# Our libs
from datasets.mit_sceneparsing_benchmark_dataset import MITSceneParsingBenchmarkDataset
from utils.utils import AverageMeter, colorEncode, accuracy , intersectionAndUnion, as_numpy
from models import get_model

#from utils.loss import CrossEntropy2d
# from utils.loss import CrossEntropyLoss2d
#from datasets.mit_dataset import MITSceneParsingDataset

#start_time = timeit.default_timer()



# RESTORE_FROM = './checkpoints/.pth'
RESTORE_FROM = './checkpoints/ADE2016-DeepLab50-ngpus2-batchSize16-h352-w352-lr0.003-epoch100-BN/DeepLab50_epoch100.pth'



start_time = timeit.default_timer()

# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################################################
    
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


def evaluate(models, val_loader, interp, criterion, args):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()
    
    models.eval()
    
    for i, batch_data in enumerate(val_loader):
        # forward pass
        images, labels, _ = batch_data
        
        torch.cuda.synchronize()
        tic = time.perf_counter()
         
        pred_seg = torch.zeros(images.size(0), args.num_classes, labels.size(1), labels.size(2))
        pred_seg = pred_seg.cuda(args.gpu_id, non_blocking=True)
        
        for scale in args.scales:
            imgs_scale = zoom(images.numpy(),
                              (1., 1., scale, scale),
                              order=1,
                              prefilter=False,
                              mode='nearest')
            
            input_images = torch.from_numpy(imgs_scale)
            if args.gpu_id is not None:
                input_images = input_images.cuda(args.gpu_id, non_blocking=True)
            
            pred_scale, _ = models(input_images)   # change
            pred_scale = interp(pred_scale)
            
            # average the probability
            pred_seg = pred_seg + pred_scale / len(args.scales)
            
        # pred =torch.log(pred)
        
        seg_labels = labels.cuda(args.gpu_id, non_blocking=True)
        
        loss = criterion(pred_seg, seg_labels)
        loss_meter.update(loss.data.item())
        print('[Eval] iter {}, loss: {}'.format(i, loss.data.item()))
        # loss_meter.update(loss.item())
        # print('[Eval] iter {}, loss: {}'.format(i, loss.item()))
        
        labels = as_numpy(labels)
        _, pred = torch.max(pred_seg, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())
        
        # calculate accuracy
        acc, pix = accuracy(pred, labels)
        intersection, union = intersectionAndUnion(pred, labels, args.num_classes)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        
        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)
        
        if args.visualize:
            visualize_result(batch_data, pred_seg, args)
            
    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [ {} ], IoU: {:.4f}'.format(i, _iou))
            
    print('[Eval Summary]:')
    print('loss: {:.6f}, Mean IoU: {:.2f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(loss_meter.average(), iou.mean() * 100, acc_meter.average()*100, time_meter.average()))
        
    
def main(args):
    torch.cuda.set_device(args.gpu_id)
    
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
    
    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = torch.nn.Upsample(size=(args.imgSize[0], args.imgSize[1]), mode='bilinear', align_corners=True)
    else:
        interp = torch.nn.Upsample(size=(args.imgSize[0], args.imgSize[1]), mode='bilinear')
        # interp = F.interpolate(size=(args.imgSize[0], args.imgSize[1]), mode='bilinear') # umsample
    
    # Dataset and Loader
    val_dataset = MITSceneParsingBenchmarkDataset(root=args.root_dataset, split="validation", img_size=args.imgSize, max_sample=args.num_val)
    val_loader = data.DataLoader(val_dataset,
                                  batch_size =int(args.batch_size) , # args.batch_size 6
                                  shuffle = False,
                                  num_workers = int(args.workers),
                                  drop_last = True)
    
    # feed into GPU
    model.cuda()    
    
    # Main loop
    evaluate(model, val_loader, interp, crit, args)
            
    end_time = timeit.default_timer()
    print('Running time(h): [{0:.4f}]h'.format((end_time - start_time) / 3600))
    print('Evaluate Done!   ')


if __name__ == '__main__':
    """
    return: A list of parsed arguments
    """
    parser = argparse.ArgumentParser(description="AttentionSegmentationModel")
    # Model related arguments
    parser.add_argument("--model", type=str, default='ASPNet50', 
                        help="Availabel options: ASPNet50 ...")
    parser.add_argument("--id", type=str, default='ADE2016', 
                        help="A name for identifying the model.")
    
    # optimization related arguments
    parser.add_argument('--num_gpus', type=int, default=1, 
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu', type=int, default=1, 
                        help='input batch size')
    parser.add_argument('--gpu_id', type=int, default=0,  # None 0 1
                        help='gpu_id for evaluation')
    
    
    # Data related arguments 
    parser.add_argument('--num_classes', type=int, default=150, 
                        help='number of classes')
    parser.add_argument('--imgSize', type=tuple, default=(384, 384),  # (512, 683) (224, 224)
                        help='input image size')
    parser.add_argument('--workers', type=int, default=6, 
                        help='number of data loading workers')
    parser.add_argument('--num_val', type=int, default= -1,   # 128
                        help='number of images to evalutate')
    
    # path related arguments
    parser.add_argument("--root-dataset", type=str, default="./data/ADEChallengeData2016/",
                        help="With all your images in an images/directory off the root of your site")
    parser.add_argument("--restore-from", type=str, default= RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--result", type=str, default="./result/eval",
                        help="Folder to output visualization results.")
    
    # Misc arguments
    parser.add_argument("--visualize", type=int, default=1,
                        help="Output visualation: 0 or 1")
       
    
    args = parser.parse_args()
    
    args.scales = (0.5, 0.75, 1, 1.25, 1.5)
    
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.result = os.path.join(args.result, args.model)
    
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    
    main(args)
    
    #model = get_model(name=args.model, num_classes = args.num_classes)
    #x = torch.randn([2, 3, 512, 512])
    #model(x)
    #print(model)
    #pass
       

