
# coding: utf-8

from mxnet import autograd
from mxnet import ndarray as nd
import numpy as np


def SGD(params, lr):
    for param in params:
        # param[:] 可以覆盖原内存更新值，不需要开辟新的存储空间
        param[:] = param - lr * param.grad

def accuracy(output, label):
    return nd.mean(output.argmax(axis= 1) == label).asscalar()

def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)