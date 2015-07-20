#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F


x = Variable(np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32))
em = F.EmbedID(8, 3)
H=2
W=4

W = np.random.randn(8, 3).astype(np.float32)
p = x.data

print x.data
print W[p]
print W[p].reshape(2, 12)


