#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:21:31 2018

@author: sunshine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index, reduce=True)
        
    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=-100):
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index, reduce=True)
        
    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)