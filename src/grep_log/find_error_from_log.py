#!/usr/bin/env python
# coding: utf-8
import os
import re
import sys

__author__ = 'k_morishita'

"""
問題： Logファイルの各行から /apple/i を含む行を見つける、ということを学習できるか。
入力： 1行は100文字以下からなるとする。1文字は [char byte code]/255 という実数で表される。 （入力層は 100個）
出力： 正規表現の /apple/i がマッチする行のときにy2=1とする。何もマッチしないならy1=1とする。 （出力層は２個）

log.txt は 1000行で、 /apple/i にマッチする行は135行ある。Log.txtの90％を学習データ、残り10％を評価データとする。
"""

LINE_MAX_CHAR = 100

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

def generate_cases(log_file):
    all_data = []
    pattern = re.compile("apple", re.I)

    with file(log_file) as f:
        for line in f:
            # INPUT
            data = [ord(x)/255.0 for x in line[:LINE_MAX_CHAR]]
            data += [0] * (100 - len(data))  # Padding
            # OUTPUT
            data += [1 if pattern.search(line) else 0]
            all_data.append(data)
    np.random.seed(1)  # データの分け方を固定
    np.random.shuffle(all_data)
    num_train = int(len(all_data) * 0.9)

    x_train = np.array([xx[:-1] for xx in all_data[:num_train]], dtype=np.float32)
    y_train = np.array([xx[-1]  for xx in all_data[:num_train]], dtype=np.int32)
    x_test  = np.array([xx[:-1] for xx in all_data[num_train:]], dtype=np.float32)
    y_test  = np.array([xx[-1]  for xx in all_data[num_train:]], dtype=np.int32)
    return x_train, y_train, x_test, y_test

def main(log_file, h_sizes, improve_loss_min=0.001):
    x_train, y_train, x_test, y_test = generate_cases(log_file)

    in_size = LINE_MAX_CHAR
    out_size = 2
    layers = [in_size] + h_sizes + [out_size]
    model = FunctionSet()
    for li in range(1, len(layers)):
        setattr(model, "l%d" % li, F.Linear(layers[li-1], layers[li]))

    optimizer = optimizers.SGD()
    optimizer.setup(model.collect_parameters())
    last_loss = None
    for epoch in range(3000000):
        optimizer.zero_grads()
        loss, accuracy = forward(model, x_train, y_train)
        loss.backward()

        if epoch % 100 == 0:
            print "epoch: %s, loss: %s, accuracy: %s" % (epoch, loss.data, accuracy.data)
            if last_loss is not None and last_loss - improve_loss_min < loss.data:
                print "Finish Training"
                break
            last_loss = loss.data

        optimizer.update()
        if epoch % 1000 == 0:
            loss, accuracy = forward(model, x_test, y_test)
            print "epoch: %s, Try Test Result: loss: %s, accuracy: %s" % (epoch, loss.data, accuracy.data)

    # result
    loss, accuracy = forward(model, x_test, y_test)
    print "epoch: %s, Test Result: loss: %s, accuracy: %s" % (epoch, loss.data, accuracy.data)
    return epoch, accuracy.data

if __name__ == '__main__':
    filename = sys.argv[1]
    if not os.path.exists(filename):
        raise "Can not read file '%s'" % filename
    main(filename, [100, 20], improve_loss_min=0.001)

