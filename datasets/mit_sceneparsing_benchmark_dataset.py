#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 21:21:28 2018

@author: sunshine
"""

import os
#import math
import random
import collections
import numpy as np
import scipy.misc as m

import cv2
import torch
import torch.utils.data as torchdata
import torchvision
from torchvision import transforms
#from skimage import transform

import matplotlib.pyplot as plt
#from ptsemseg.utils import recursive_glob

def recursive_glob(rootdir=".", suffix=""):
    """
    Performs recursive glob with given suffix and rootdir
    : param rootdir is the root directory
    : param suffix is the suffix to be searched
    
    # chinese
    列表生成式，可能占用过多的内存
    [ 要放入列表的数据   简单的表达式1  表达式2 ]
    x for x in range(0, index)   for循环遍历出来的值,放入列表中
    
    """
    
    return [
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
            ]  

class MITSceneParsingBenchmarkDataset(torchdata.Dataset):
    """MITSceneParsingBenchmarkLoader

    http://sceneparsing.csail.mit.edu/

    Data is derived from ADE20k, and can be downloaded from here:
    http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

    NOTE: this loader is not designed to work with the original ADE20k dataset;
    for that you will need the ADE20kLoader

    This class can also be extended to load data for places challenge:
    https://github.com/CSAILVision/placeschallenge/tree/master/sceneparsing
    
    "__len__" : that provides the size of the dataset,
    "__getitem__": supporting integer indexing in range from 0 to len(self) exclusive 
    
    # chinese
    "__init__": 得到图像的路径，然后将图像路径组成一个数组。一些初始化过程写在这里。
    "__len__" : 返回数据集的大小。
    "__getitem__": 通过索引返回数据（图像）和标签。注意：在pytorch中得到的图像必须是tensor。

    """
    def __init__(self, root, split="training",  img_size=512, max_sample=-1, augmentations=None, is_transform=False, img_norm=True):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 150  # 0 is reserved for "other"
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        #self.seg_size = seg_size if isinstance(seg_size, tuple) else (seg_size, seg_size)
        
        # mean and std
        # mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.]  # img_norm = False
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # img_norm = True
        self.img_transfrom = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    
        #self.mean = np.array([104.00699, 116.66877, 122.67892])
        #self.files = {}
        
        self.images_base = os.path.join(self.root, 'images', self.split)
        self.annotations_base = os.path.join(self.root, 'annotations', self.split)
        
        #self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.jpg')
        self.files = collections.defaultdict(list)
        for split in ["training", "validation",]:
            file_list = recursive_glob(rootdir=self.images_base, suffix='.jpg')
            self.files[split] = file_list
        
        if split == "training" :
            random.shuffle(self.files[split])
        
        if max_sample > 0:
            self.files[split] = self.files[split][0 : max_sample]

        if not self.files[split]:
            raise Exception("No files for split=[{}] found in {}".format(split, self.images_base))
            
        self.num_sample = len(self.files[split])
        assert self.num_sample > 0
        print('Found samples : {} | {} images'.format(self.num_sample, self.split))

    def __len__(self):
        """__len__  返回数据集的大小
        """
        return len(self.files[self.split])
    
    def _scale_and_corp(self, img, seg, cropSize, split):
        h, w = img.shape[0], img.shape[1]
        
        if split == "training":
            # random scale
            scale = random.random() + 0.5 # 0.5-1.5
            scale = max(scale, 1. * min(cropSize[0], cropSize[1]) / (min(h, w)-1))
        else:
            # scale to crop size
            scale = 1. * min(cropSize[0], cropSize[1]) / (min(h, w)-1)
        
        img_scale = m.imresize(img, scale, interp='bilinear')
        seg_scale = m.imresize(seg, scale, interp='nearest')
        # seg_scale = m.imresize(seg, scale, interp='nearest', mode='F')
        
        h_s, w_s = img_scale.shape[0],  img_scale.shape[1]
        
        if split == "training":
            # random crop
            x1 = random.randint(0, w_s - cropSize[1])
            y1 = random.randint(0, h_s - cropSize[0])
        else:
            x1 = (w_s - cropSize[1]) // 2
            y1 = (h_s - cropSize[0]) // 2
            
        img_crop = img_scale[y1 : y1 + cropSize[0], x1 : x1 + cropSize[1], : ]
        seg_crop = seg_scale[y1 : y1 + cropSize[0], x1 : x1 + cropSize[1] ]
        
        return img_crop, seg_crop
            
    def _flip(self, img, label):
        img_flip = img[:, ::-1, :]
        label_flip = label[:, ::-1]
        
        return img_flip, label_flip

    def __getitem__(self, index):
        """__getitem__

        :param index:
            
        rstrip() : 删除字符串末尾的指定字符(默认为空格).
        os.path.basename(): 返回path最后的文件名. 如果path以/或\结尾，则返回空值.
        """
        img_path = self.files[self.split][index].rstrip()
        label_path = os.path.join(self.annotations_base, os.path.basename(img_path)[:-4] + '.png')
        #print(img_path)
        #print(label_path)
        
        # assert whether path exists 断言路径是否存在
        assert os.path.exists(img_path), '[{}] does not exist'.format(img_path)
        assert os.path.exists(label_path), '[{}] does not exist'.format(label_path)
        
        # load image and label
        try:
            img = m.imread(img_path, mode='RGB')
            #img = np.array(img, dtype=np.uint8)
            #img = img[:, :, ::-1]  # RGB --> BGR !!!
            #print(img.shape)  #  eg. (512, 512, 3)
            label = m.imread(label_path)
            #label = np.array(label, dtype=np.uint8)
            #print(label.shape)  #  eg. (512, 512)
            
            assert(img.ndim == 3)
            assert(label.ndim == 2)
            assert(img.shape[0] == label.shape[0])
            assert(img.shape[1] == label.shape[1])
            
            # image to float
            # img = img[:, :, ::-1]  # RGB --> BGR !!!
            #img = img.astype(np.float32)[:, :, ::-1]  # 64
            #label = label.astype(np.float32)
            
            # classes = np.unique(label)  
            # scale and crop
            # flip
            if self.split == "training":
                random_flip = np.random.choice([-1, 0, 1, 2])
                # print(random_flip)
                if random_flip == 1:
                    #img, label = self._flip(img, label)
                    img = cv2.flip(img, 1)
                    label = cv2.flip(label, 1) 
                elif random_flip == 0:
                    img = cv2.flip(img, 0)
                    label = cv2.flip(label, 0)
                else:
                    img = img.copy()
                    label = label.copy()
                    
                """
                elif random_flip == -1:
                    img = cv2.flip(img, -1)
                    label = cv2.flip(label, -1)
                """
                
            if self.img_size[0] > 0 and self.img_size[1] > 0:
                img, label = self._scale_and_corp(img, label, self.img_size, self.split)
                
            # order默认是1,双线性. resize后image的范围又变为[0 - 1]
            # img = transform.resize(img, (self.img_size[0], self.img_size[1]), order=1) 
            
            # img = m.imresize(img, (self.img_size[0], self.img_size[1]), interp='bilinear') # uint8 with RGB mode
            # img = cv2.resize(img.copy(), (self.img_size[0], self.img_size[1])) 
            
            # image to float
            # image_ori = img.copy()  # original image
            # img = img.astype(np.float32)  # 64
            
            img = img.astype(np.float32)[:, :, ::-1]    # RGB --> BGR !!!
            # label = label.astype(np.float32)
            
            if self.img_norm:
                # Resize scales images from 0 to 255, thus we need to divide by 255.0
                img = img.astype(np.float32) / 255.0
            img = img.transpose((2, 0, 1))  # NHWC --> NCHW
            
            # classes = np.unique(label)    # np.unique() :对于一维数组或列表,函数去除其中的重复元素,并按元素又小到大返回一个新的无元素重复的数组或列表
            # label = label.astype(np.float64)
            # label = m.imresize(label, (self.img_size[0], self.img_size[1]), interp='nearest', mode='F')
            # print(classes)
            # print('label', np.unique(label))
            
            # label = m.imresize(label, (self.img_size[0], self.img_size[1]), interp='nearest')
            
            # label to int from 0/-1 to 150/149 totall 151
            label = label.astype(np.int) - 1
            # label = label.astype(np.int)
            
            """
            if not np.all(classes == np.unique(label)):
                print("WARN: resizing labels yielded fewer classes")
            """
            if not np.all(np.unique(label) < self.n_classes):
                raise ValueError("Segmentation map contained invalid class values")
            
            if self.augmentations is not None:
                img, label = self.augmentations(img, label)
            
            if self.is_transform:
                img, label = self.transform(img, label)
            
            # to torch tensor
            image = torch.from_numpy(img)
            # segmentation = torch.from_numpy(label)
            segmentation = torch.from_numpy(label).long()
            
            # segmentation = segmentation - 1
            
            # substracted by mean and divided by std
            # image = self.img_transfrom(image)
            
        except Exception as e:
            print('Failed loading image/label [{}]: {}'.format(img_path, e))
            
            # dummy datw
            
            image = torch.zero(3, self.img_size[0], self.img_size[1]) # (C, H, W)
            segmentation = -1 * torch.ones(self.img_size[0], self.img_size[1]).long()
            
            return image, segmentation
        
        # substracted by mean and divided by std
        image = self.img_transfrom(image)
        
        return  image, segmentation, img_path

            
    
if __name__ == '__main__':
    local_path = '/home/hi/BOBO/AttentionSegModel/data/ADEChallengeData2016/'
    dst = MITSceneParsingBenchmarkDataset(local_path,  split="training")
    trainloader = torchdata.DataLoader(dst, 
                                  batch_size=4,
                                  shuffle=False,
                                  num_workers=int(16),
                                  pin_memory=True,
                                  drop_last=True)
    for i, datas in enumerate(trainloader):
        imgs, labels, path = datas
        print(path)
        if i == 1:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            #img = img_ori.numpy()
            ##img = torchvision.utils.make_grid(img_ori).numpy()
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            break
            #for j in range(4):
                #plt.imshow(dst.decode_segmap(labels.numpy()[j]))
                #plt.show()
            #break