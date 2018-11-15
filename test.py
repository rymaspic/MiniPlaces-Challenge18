from __future__ import print_function
from __future__ import division

import torch
import torchvision.transforms as transforms
import os

from scipy.misc import imread, imresize

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

def load_imagepaths_from_folder(folder):
    paths = []
    for filename in os.listdir(folder):
        path = os.path.join(folder,filename)
        if path is not None:
            paths.append(path)

    return sorted(paths)


def construct_transformer():
    """construct transformer for images"""
    mean = [0.45486851, 0.43632515, 0.40461355]
    std = [0.26440552, 0.26142306, 0.27963778]
    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transformer

def main():
    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model and set to evaluation mode
    model = load_model('ResNet')
    model.to(device)
    model.eval()
    # set image transformer
    transformer = construct_transformer()
    paths = load_imagepaths_from_folder('data/test/999/')
    # load the image
    f = open("model3.txt", "w")  # opens file with name of "test.txt"
    for path in paths:
        image = imread(path)
        image = transformer(image)
        image = image.view(-1, 3, 128, 128)
        image = image.to(device)
        # run the forward process
        prediction = model(image)
        prediction = prediction.to('cpu')
        _, cls = torch.topk(prediction, dim=1, k=5)
        output = path
        for i in cls.data[0]:
            output = output + " "
            output = output + str(i.item())
        a = output.split("/")
        output = a[1] + "/" + a[3]
        f.write(output + "\n")
        #os.system("echo %s > text1/output_file.txt" %output)
        x = input()
    
    f.close()

main()
