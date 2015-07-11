#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

"""
N bit の ２進数で表される 2**N 個の数字を 2**N個のOutputのどれかを１にするということを学習させる。
"""

import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F


def forward(model, x_data, y_data):
    x = Variable(x_data)
    t = Variable(y_data)
    if hasattr(model, "l2"):
        h1 = model.l1(x)
        y = model.l2(h1)
    else:
        y = model.l1(x)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def generate_training_cases(n_bit):
    x_data = []
    t_data = []
    output_len = 2**n_bit
    for i in range(output_len):
        x_data.append(list((int(x) for x in ("0"*n_bit + bin(i)[2:])[-n_bit:])))
        t_data.append(i)
    return np.array(x_data, dtype=np.float32), np.array(t_data, dtype=np.int32)

def main(n_bit, h1_size):
    if h1_size > 0:
        model = FunctionSet(
            l1=F.Linear(n_bit, h1_size),
            l2=F.Linear(h1_size, 2**n_bit)
        )
    else:
        model = FunctionSet(
            l1=F.Linear(n_bit, 2**n_bit)
        )
    optimizer = optimizers.SGD()
    optimizer.setup(model.collect_parameters())
    x_data, t_data = generate_training_cases(n_bit)
    for epoch in range(100000):
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
    fp = file("result.txt", "w")
    fp.write("N\tH1\tepoch\taccuracy\n")
    for n_bit in range(4, 8):
        for h1_size in [0, n_bit, n_bit * 2, n_bit * 4, 2**n_bit]:
            epoch, accuracy = main(n_bit, h1_size)
            fp.write("%s\t%s\t%s\t%s\n" % (n_bit, h1_size, epoch, accuracy))
