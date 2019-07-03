#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:44:10 2019

@author: hi
"""

import torch
import torch.nn as nn
m = nn.Softmax2d() # dim=1
n = nn.Softmax() # dim=1
n2 = nn.Softmax(dim=2)
n3 = nn.Softmax(dim=3)
s = nn.Sigmoid()
# you softmax over the 2nd dimension
input = torch.randn(2, 1, 4, 4)
print(input)

outs = s(input)
print(outs)

output = m(input)
print(output)

out1 = n(input)
print(out1)
out2 = n2(input)
print(out2)
out3 = n3(input)
print(out2)

out = out2 + out2
print(out)


Fo =torch.Tensor([[[[0.9, 0.9, -0.7, 0.3, -0.3],
                    [0.9, 0.9, 0.9, 0.9, 0.9],
                    [0.5, 0.5, 0.5, 0.5, 0.5],
                    [0.2, 0.5, 0.0, 0.7, -0.4],
                    [0.4, 0.1, 0.6, -0.1, 0.8]
                    ]]])

# Fo = torch.from_numpy(Fo)
print(Fo.size())
print(Fo.shape)
Fs = s(Fo)
print(Fs)
Fh = n2(Fo)
print(Fh)
Fw = n3(Fo)
print(Fw)
