#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F


model = FunctionSet(
    l1=F.Linear(4, 3),
    l2=F.Linear(3, 2),
    l3=F.Linear(2, 2)
)

x = Variable(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32))
h1 = model.l1(x)
h2 = model.l2(h1)
h3 = model.l3(h2)

optimizer = optimizers.SGD()
optimizer.setup(model.collect_parameters())
optimizer.zero_grads()

xx = Variable(np.array([[1,2,3,4], [0, 1, 0.5, 0.2]], dtype=np.float32))
print F.accuracy(xx, Variable(np.array([3, 1], dtype=np.int32))).data

