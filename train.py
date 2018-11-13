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

def calcTopKError(loader, k, name):
    epoch_topk_err = 0.0

    for batch_num, (inputs, labels) in enumerate(loader, 1):
        inputs = inputs.to(device)
        labels = labels.to(device)

        _,cls = torch.topk(outputs,dim=1,k=k)
        batch_topk_err = (1 - (cls.numel()-torch.nonzero(cls-labels.view(-1,1)).shape[0])/labels.numel())
        epoch_topk_err += batch_topk_err

        if batch_num % output_period == 0:
            # print('[%d:%.2f] %s_Topk_error: %.3f' % (
            #     epoch,
            #     batch_num*1.0/num_val_batches,
            #     name,
            #     epoch_topk_err/batch_num
            # ))
            gc.collect()

    return epoch_topk_err

def run():
    train_top1 = []
    train_top5 = []
    val_top1 = []
    val_top5 = []

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
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
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
            #print(labels.shape)
            #print(inputs.shape)
            if (batch_num == 100):
                break
                
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs)
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

        model.eval()

        # TODO: Calculate classification error and Top-5 Error
        def calcHelper(name, k):
            return calcTopKError(eval(name+"_loader"), k, name)

        train_top1_err = calcHelper("train", 1)
        train_top5_err = calcHelper("train", 5)
        val_top1_err =  calcHelper("val", 1)
        val_top5_err = calcHelper("val", 5)

        gc.collect()

        train_top1.append(train_top1_err)
        train_top5.append(train_top5_err)
        val_top1.append(val_top1_err)
        val_top5.append(val_top5_err)
        print("Train_Top1_loss in Epoch" + str(epoch) + ": " + str(train_top1_err))
        print("Train_Top5_loss in Epoch" + str(epoch) + ": " + str(train_top5_err))
        print("Val_Top1_loss in Epoch" + str(epoch) + ": " + str(val_top1_err))
        print("Val_Top5_loss in Epoch" + str(epoch) + ": " + str(val_top5_err))

        epoch += 1

    x_idx = range(num_epochs)
    plt.plot(x_idx, train_top1, label="train_top1")
    plt.plot(x_idx, train_top5, label="train_top5")
    plt.plot(x_idx, val_top1, label="val_top1")
    plt.plot(x_idx, val_top5, label="val_top5")
    plt.savefig("res.pdf")

print('Starting training')
run()
print('Training terminated')
