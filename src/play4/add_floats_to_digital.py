#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

"""
問題： 0~Nの整数を0~1の実数にMappingした入力を２つとり、その足し算を0~2Nの対応するNodeに出力することを学習できるか
入力： ２つの k1/N, k2/N (k1,k2 = 0~N) という実数 (入力層は2個)
出力： k1+k2 に対応する Nodeを１にする (出力層は 2N+1個)
"""

import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

def forward(model, x_data, y_data):
    x = Variable(x_data)
    t = Variable(y_data)
    for i in range(1, 1000):  # 1000 は適当な数
        if hasattr(model, "l%d" % i):
            x = getattr(model, "l%d" % i)(x)
        else:
            y = x
            break
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def generate_training_cases(max_n):
    x_data = []
    t_data = []
    for k1 in range(max_n+1):
        for k2 in range(max_n+1):
            x_data.append([1.0*k1 / max_n, 1.0*k2/max_n])
            t_data.append(k1+k2)
    return np.array(x_data, dtype=np.float32), np.array(t_data, dtype=np.int32)

def main(max_n, h_sizes):
    x_data, t_data = generate_training_cases(max_n)

    in_size = 2
    out_size = 2*max_n+1
    layers = [in_size] + h_sizes + [out_size]
    model = FunctionSet()
    for li in range(1, len(layers)):
        setattr(model, "l%d" % li, F.Linear(layers[li-1], layers[li]))

    optimizer = optimizers.SGD()
    optimizer.setup(model.collect_parameters())
    for epoch in range(3000000):
        optimizer.zero_grads()
        loss, accuracy = forward(model, x_data, t_data)
        loss.backward()
        if epoch % 100 == 0:
            print "epoch: %s, loss: %s, accuracy: %s" % (epoch, loss.data, accuracy.data)
        if accuracy.data == 1:
            break
        optimizer.update()
    print "epoch: %s, loss: %s, accuracy: %s" % (epoch, loss.data, accuracy.data)
    return epoch, accuracy.data

if __name__ == '__main__':
    main(8, [8])

