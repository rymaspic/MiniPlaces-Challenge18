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

def load_model(model_name):
    """load the pre-trained model"""
    if model_name == 'ResNet':
        model = resnet_18()
        model_path = './models/model_Pro2.20'
    elif model_name == 'AlexNet':
        model = alexnet()
        model_path = './models/alexnet.pt'
    else:
        raise NotImplementedError(model_name + ' is not implemented here')

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model

def run():
    train_top1 = []
    train_top5 = []
    val_top1 = []
    val_top5 = []

    # Parameters
    num_epochs = 20
    output_period = 10
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model('ResNet')
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
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
        torch.save(model.state_dict(), "models/model_Pro2.%d" % epoch+20)

        model.eval()

        # a helper function to calc topk error
        def calcTopKError(loader, k, name):
            epoch_topk_err = 0.0

            for batch_num, (inputs, labels) in enumerate(loader, 1):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                _,cls = torch.topk(outputs,dim=1,k=k)
                batch_topk_err = (1 - (cls.numel()-torch.nonzero(cls-labels.view(-1,1)).shape[0])/labels.numel())
                epoch_topk_err = epoch_topk_err * ((batch_num-1) / batch_num) \
                                + batch_topk_err / batch_num

                if batch_num % output_period == 0:
                    # print('[%d:%.2f] %s_Topk_error: %.3f' % (
                    #     epoch,
                    #     batch_num*1.0/num_val_batches,
                    #     name,
                    #     epoch_topk_err/batch_num
                    # ))
                    gc.collect()


            return epoch_topk_err

        # TODO: Calculate classification error and Top-5 Error
        train_top1_err = calcTopKError(train_loader, 1, "train")
        train_top5_err = calcTopKError(train_loader, 5, "train")
        val_top1_err =  calcTopKError(val_loader, 1, "val")
        val_top5_err = calcTopKError(val_loader, 5, "val")

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
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.plot(x_idx, train_top1, label="train_top1")
    plt.plot(x_idx, train_top5, label="train_top5", linestyle="--")
    plt.plot(x_idx, val_top1, label="val_top1")
    plt.plot(x_idx, val_top5, label="val_top5", linestyle="--")
    plt.legend()
    plt.savefig("res_Pro2_ADAM_NO_DROP_RE.pdf")

print('Starting training')
run()
print('Training terminated')
