#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from src.sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from src.sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    返回训练和测试集，和一个用户组（字典，key是用户位置，value是相对应的数据）
    """

    if args.dataset == 'cifar':  # 取cifar数据集
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),  # 将【0，255】的值转换成shape为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 均值、方差，正则化，Normalized_image=(image-mean)/std

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users 用户抽取训练数据
        if args.iid:  # 抽取独立同分布的数据
            # Sample IID user data from cifar
            user_groups = cifar_iid(train_dataset, args.num_users)  # 字典，key是用户编号比如1到10，value是一个set，是数据集中的数据的编号，如 3，14，17...
        else:
            # Sample Non-IID user data from cifar   抽取非独立同分布的数据
            if args.unequal:
                # Chose uneuqal splits for every user  非独立同分布的不均等的分割 对于cifar没有不等
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user    非独立同分布的均等分割
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))  # 除法 求模型权重的均值
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
