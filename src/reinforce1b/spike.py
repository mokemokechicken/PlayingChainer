#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'


import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

from random import random, choice, randint

def forward(model, state):
    x = Variable(np.array([[state]], dtype=np.float32))
    for i in range(1, 1000):  # 1000 は適当な数
        if hasattr(model, "l%d" % i):
            x = getattr(model, "l%d" % i)(x)
        else:
            y = x
            break
    return y


model = FunctionSet(
    l1=F.Linear(1, 10),
    l2=F.Linear(10, 4),
)
optimizer = optimizers.SGD()
optimizer.setup(model.collect_parameters())


optimizer.zero_grads()
q_last = forward(model, 1/9.0)
tt = np.copy(q_last.data)
tt[0][2] = 0.3
target = Variable(tt)

loss = F.mean_squared_error(q_last, target)
loss.backward()





print dir(q_last)


