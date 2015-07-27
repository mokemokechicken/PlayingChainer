#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

model = FunctionSet(
    l1=F.Linear(1000, 10)

)
# model.parameters = (np.array([[1]], dtype=np.float32), np.array([[0]], dtype=np.float32))
optimizer = optimizers.SGD()
optimizer.setup(model.collect_parameters())

def forward(model, value, target, volatile):
    x = Variable(np.array([[value]], dtype=np.float32), volatile=volatile)
    y = model.l1(x)
    loss = 0.5 * (y - target) ** 2
    return loss

def update(volatile):
    optimizer.zero_grads()
    loss = 0
    loss += forward(model, [1]*1000, np.array([4]*10), volatile=volatile)
    # loss.backward()
    optimizer.update()


import timeit

print timeit.timeit("update(False)", setup="from __main__ import update", number=1000)
print timeit.timeit("update(True)", setup="from __main__ import update", number=1000)


