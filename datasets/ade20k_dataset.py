#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 21:19:02 2018

@author: sunshine
"""

# ADE20K dataset loader

import os
import collections
import torch
import torchvision

import numpy as np
import scipy.misc as m

#import skimage.io as io
#import skimage.transform as trans
import matplotlib.pyplot as plt

from torch.utils import data
#from skimage import io, transform

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
    
    
class ADE20KDataset(data.Dataset):
    """
    "__len__" : that provides the size of the dataset,
    "__getitem__": supporting integer indexing in range from 0 to len(self) exclusive 
    
    # chinese
    "__init__": 得到图像的路径，然后将图像路径组成一个数组。一些初始化过程写在这里。
    "__len__" : 返回数据集的大小。
    "__getitem__": 通过索引返回数据（图像）和标签。注意：在pytorch中得到的图像必须是tensor。
    
    """
    
    def __init__(self, root, split="training", is_transform=False, img_size=512, augmentations=None, img_norm=True):  
        """
        这个列表list存放所有图像的地址
        root: 图像存放地址根路径
        augmentations: 是否需要图像增强
        
        #####################
        tuple() : 元组
        isinstance(object, classes)  : 判断一个对象是否是已知的类型
        
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 150
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        
        for split in ["training", "validation",]:
            file_list = recursive_glob(rootdir=self.root + 'images/' + self.split + '/', suffix='.jpg')  
            self.files[split] = file_list
            #print(self.files)
            
        
    def __len__(self):
        # 返回数据集的大小
        return len(self.files[self.split])
    
        
    def __getitem__(self, index):
        """
        rstrip() : 删除字符串末尾的指定字符(默认为空格)
        """
        img_path = self.files[self.split][index].rstrip()
        label_path = img_path[:-4] + '_seg.png'
        # assert whether path exists 断言路径是否存在
        assert os.path.exists(img_path), '[{}] does not exist'.format(img_path)
        assert os.path.exists(label_path), '[{}] does not exist'.format(label_path)
        
        #if self.img_size[0] > 0:
            #print('aaaa', self.img_size[0]) 
    
            
        # load image and label
        # scipy1.0.0 -> scipy1.2.0 : imresize --> skimage.transform.resize
        try:
            img = m.imread(img_path, mode='RGB')
            #img = io.imread(img_path, as_gray=False) #  read a RGB image
            #print(img.ndim)
            #assert(img.ndim == 3)
            img = np.array(img, dtype=np.uint8)
        
            label = m.imread(label_path)
            #label = io.imread(label_path)  # read a RGB image
            #print(label.ndim)
            #assert(label.ndim == 3)
            label = np.array(label, dtype=np.int32)
            
        except Exception as e:
            print('Failed loading image/label [{}]: {}'.format(img_path, e))
        
        if self.augmentations is not None:
            img, label = self.augmentations(img, label)
            
        if self.is_transform:
            img, label = self.transform(img, label)
            
        return img, label
    
    def transform(self, img, label):
        """
        np.unique() :对于一维数组或列表,函数去除其中的重复元素,
                    并按元素又小到大返回一个新的无元素重复的数组或列表
                    
        
        """
        
        img = m.imresize(img, (self.img_size[0], self.img_size[1]), interp='bilinear') # uint8 with RGB mode
        #img = trans.resize(img, (self.img_size[0], self.img_size[1]))
        img = img[:, :, ::-1]  # RGB -->  BGR  
        img = img.astype(np.float64)
        img -= self.mean  # print(img.mean())
        
        if self.img_norm:
            # Resize scales images from o to 255, thus we need to divide by 255.0
            # image to float
            img = img.astype(float) / 255.0
            
        # RGB: R - 0, G - 1, B - 2
        #print(img.shape)  # shape: (512, 512, 3)  (height, width, channel)
        
        # NHWC --> NCHW
        img = img.transpose(2, 0, 1)
       
        # 图像尺寸
        # 图像总像素个数
        # 图像类型
        #print(img.shape)  # shape: (3, 512, 512)  (channel, height, width)
        #print(img.size)   # 786432
        #print(type(img))  # numpy.ndarray
        
        
        label = self.encode_segmap(label)
        #print(label)
        classes = np.unique(label)
        #print(classes)
        label = label.astype(float)   # 
        #print(label)
        label = m.imresize(label, (self.img_size[0], self.img_size[1]), interp='nearest', mode='F')
        #label = trans.resize(label, (self.img_size[0], self.img_size[1])) 
        label = label.astype(int)     # label to int from 0 to 150  totall 151
        #print(label)
        assert(np.all(classes == np.unique(label)))  # np.all  -> and
        
        # to torch tensor
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()
        
        return img, label
    
    
    def encode_segmap(self, mask):
        # Refer: http://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20k.m
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]))
        label_mask = ( mask[:, :, 0] / 10.0 )*256 + mask[:, :, 1]
        return np.array(label_mask, dtype=np.uint8)
    
    def decode_segmap(self, temp, plot=False):
        # TODO: (@meetshah1995)
        # Verify that the color mapping is 1-to-1
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        
        for i in range(0, self.n_classes):
            r[temp == i] = 10*(1%10)
            g[temp == i] = 1
            b[temp == i] = 0
            
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = (r/255.0)
        rgb[:, :, 1] = (g/255.0)
        rgb[:, :, 2] = (b/255.0)
        
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
        
        
if __name__ == '__main__':
    local_path = '/home/sunshine/AttentionSegModel/data/ADE20K_2016_07_26/'
    dst = ADE20KDataset(local_path,  split="training", is_transform=True)
    trainloader = data.DataLoader(dst, 
                                  batch_size=4,
                                  shuffle=True,
                                  num_workers=int(16),
                                  pin_memory=True,
                                  drop_last=True)
    for i, datas in enumerate(trainloader):
        imgs, labels = datas
        #print(i)
        
        if i == 1:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            for j in range(4):
                plt.imshow(dst.decode_segmap(labels.numpy()[j]))
                plt.show()
            break
            
        
        
        
    
    
    
    
    
    
    