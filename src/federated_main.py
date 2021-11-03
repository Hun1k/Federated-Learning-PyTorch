#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from src.options import args_parser
from src.update import LocalUpdate, test_inference
from src.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from src.utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()  # 记录程序运行时间

    # define paths
    path_project = os.path.abspath('..')  # 'G:\\workspace\\Federated-Learning-PyTorch'
    logger = SummaryWriter('../logs')

    args = args_parser()  # Namespace(dataset='mnist', epochs=10, frac=0.1, gpu=None, iid=1, kernel_num=9, kernel_sizes='3,4,5', local_bs=10, local_ep=10, lr=0.01, max_pool='True', model='cnn', momentum=0.5, norm='batch_norm', num_channels=1, num_classes=10, num_filters=32, num_users=100, optimizer='sgd', seed=1, stopping_rounds=10, unequal=0, verbose=1)
    exp_details(args)  # 打印模型参数

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups 训练集 测试集 和一个各个客户端的数据分组情况， 字典类型，key：客户端 value 数据在数据集中的位置
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL  选择网络模型
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()


    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []  # 训练损失以及准确度
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):  # tqdm进度条
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')  # | Global Training Round : 1 |

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)  # 参与迭代的客户端数量
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 从客户端中随机选取m个客户端

        for idx in idxs_users:  # 对这些选中的客户端
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)  # w是此客户端经过本地训练后的权重， loss是训练的损失
            local_weights.append(copy.deepcopy(w))  # 此客户端权重加入local_weights列表
            local_losses.append(copy.deepcopy(loss))  # 此客户端损失加入local_losses列表

        # update global weights  经过上一步的客户端的本地训练，获取了所选客户端更新后的权重，现在对全局权重进行更新
        global_weights = average_weights(local_weights)

        # update global weights  更新模型
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)  # 在一次全局epoch中所选客户端的平均损失
        train_loss.append(loss_avg)  # 加入train_loss列表

        # Calculate avg training accuracy over all users at every epoch  计算每个全局epoch之后的精度与损失
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
