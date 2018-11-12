import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True
import matplotlib.pyplot as plt
import numpy as np

import dataset
from models.AlexNet import *
from models.ResNet import *

train_top1_loss_list = []
train_top5_loss_list = []
val_top1_loss_list = []
val_top5_loss_list = []
index = 0

def run():
    # Parameters
    num_epochs = 10
    output_period = 100
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch = 1
    while epoch <= num_epochs:
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            print(labels.shape)
            print(inputs.shape)

            optimizer.zero_grad()
            outputs = model(inputs)
            print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/num_train_batches,
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/model.%d" % epoch)

        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here
        train_top1_loss = 0.0
        train_top5_loss = 0.0
        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(labels.shape)

            outputs = model(inputs)
            # outputs = outputs.numpy
            _, cls = torch.max(outputs, dim=1)
            # print(torch.max(outputs,dim=1))
            # cls = cls + 1
            train_top1_loss = train_top1_loss + torch.nonzero(cls - labels).shape[0] / labels.numel()

            _, cls = torch.topk(outputs, dim=1, k=5)
            # cls = cls + 1
            train_top5_loss = train_top5_loss + (
            1 - (cls.numel() - torch.nonzero(cls - labels.view(-1, 1)).shape[0]) / labels.numel())

            if batch_num % output_period == 0:
                print('[%d:%.2f] Train_Top1_loss: %.3f Train_Top5_loss: %.3f' % (
                    epoch, batch_num * 1.0 / num_train_batches,
                    train_top1_loss / output_period,
                    train_top5_loss / output_period
                ))
                # train_top1_loss_list.append(train_top1_loss / output_period)
                # train_top5_loss_list.append(train_top5_loss / output_period)
                # index = index + 1
                train_top1_loss = 0.0
                train_top5_loss = 0.0
                gc.collect()



        val_top1_loss = 0.0
        val_top5_loss = 0.0
        for batch_num, (inputs, labels) in enumerate(val_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(labels.shape)

            outputs = model(inputs)
            #outputs = outputs.numpy
            _,cls = torch.max(outputs,dim=1)
            #print(torch.max(outputs,dim=1))
            #cls = cls + 1
            val_top1_loss = val_top1_loss + torch.nonzero(cls-labels).shape[0]/labels.numel()

            _,cls = torch.topk(outputs,dim=1,k=5)
            #cls = cls + 1
            val_top5_loss = val_top5_loss + (1 - (cls.numel()-torch.nonzero(cls-labels.view(-1,1)).shape[0])/labels.numel())

            if batch_num % output_period == 0:
                print('[%d:%.2f] Val_Top1_loss: %.3f Val_Top5_loss: %.3f' % (
                    epoch, batch_num*1.0/num_val_batches,
                    val_top1_loss/output_period,
                    val_top5_loss/output_period
                    ))
                val_top1_loss_list.append(train_top1_loss / output_period)
                val_top5_loss_list.append(train_top5_loss / output_period)

                val_top1_loss = 0.0
                val_top5_loss = 0.0
                gc.collect()

        gc.collect()
        epoch += 1

print('Starting training')
run()
for i in range(index):
    plt.plot(i,val_top1_loss_list[i],label = 'val-top1')
    plt.plot(i,val_top5_loss_list[i],label = 'val-top5')
    plt.plot(i, train_top1_loss_list[i], label='train-top1')
    plt.plot(i, train_top5_loss_list[i], label='train-top5')
plt.show()

print('Training terminated')
