#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

"""
問題：
N Bit で表される２つの２進数x1,x2の加算と減算を学習できるか.
IN: (x1, x2, add or sub), OUT: x1+x2 or x1-x2
(ただし x1 >= x2 とする)

結果：
加算の場合のみ、減算の場合のみ、なら学習できるが、
ある入力でそれを切り替えるというのは普通にやったら無理っぽい。。。？
"""

import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

ADD = 0
SUB = 1

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

def convert_num_to_n_bit_array(num, n_bit):
    return list((int(x) for x in ("0"*n_bit + bin(num)[2:])[-n_bit:]))

def generate_training_cases(n_bit):
    x_data = []
    t_data = []
    for x1 in range(2**n_bit):
        for x2 in range(x1+1):
            in_data = convert_num_to_n_bit_array(x1, n_bit) + convert_num_to_n_bit_array(x2, n_bit)
            x_data.append(in_data + [ADD])
            t_data.append(x1+x2)
            x_data.append(in_data + [SUB])
            t_data.append(x1-x2)

    return np.array(x_data, dtype=np.float32), np.array(t_data, dtype=np.int32)

def main(n_bit, h_sizes):
    in_size = n_bit+n_bit+1
    out_size = 2**(n_bit+1)
    layers = [in_size] + h_sizes + [out_size]
    model = FunctionSet()
    for li in range(1, len(layers)):
        setattr(model, "l%d" % li, F.Linear(layers[li-1], layers[li]))
    optimizer = optimizers.SGD()
    optimizer.setup(model.collect_parameters())
    x_data, t_data = generate_training_cases(n_bit)
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
    main(2, [10, 10])
