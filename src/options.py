#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")  # 联邦学习全局训练轮次
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")  # 客户端数量
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')  # 参与训练的客户端比例
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")   # 本地训练的epochs
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")  # 本地训练的batch size
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')  # 学习率
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')    # 动量随机梯度下降，对SGD的改进，减少其环绕方向的分量，向最优点行进的# 更快，更平稳 t = βt' + (1-β）t'

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')  # 模型名字
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')   # 每种核的数量
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')  # 以逗号分隔的核的大小，用于卷积
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")  # 图片的通道数量
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")  # ？不懂
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")    # omniglot具有1623个类别，但每个类别只有20张图片，即更加复杂
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")  # 数据集名字
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")    # 分类的种类数量
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")  # 是否使用GPU，数字代表GPU的序号
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")  # 优化方法，默认SGD
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')  # 1代表独立同分布，0代表非独立同分布
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')  # 不均等的数据分割（0均等）
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')  # 不懂
    parser.add_argument('--verbose', type=int, default=1, help='verbose')  # 打印日志
    parser.add_argument('--seed', type=int, default=1, help='random seed')  # 种子
    args = parser.parse_args()
    return args
