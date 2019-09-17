# Created By: Mahbub Zaman
#Sample execution command:
# python train.py --data_dir ./flowers
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from PIL import Image

import time
import datetime
import argparse


def save_checkpoint(args, model, classifier, optimizer):
    
    checkpoint = {'arch': args.arch, 
                  'model': model,
                  'classifier' : classifier,
                  'optimizer': optimizer.state_dict(),
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'epochs': args.epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    optimizer = checkpoint['optimizer']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Training Module")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg19', choices=['vgg16', 'vgg19'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.01')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='10')
    parser.add_argument('--gpu', action="store_true", default=True)
    return parser.parse_args()

def train(dataloaders, model, optimizer, criterion, epochs, gpu):
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()
    running_loss = 0
    accuracy = 0
    start = time.time()
    starttime = datetime.datetime.now()
    print('Training started at {}'.format(starttime))

    for e in range(epochs):
        trainingmode = 0
        validationmode = 1
        for mode in [trainingmode, validationmode]:   
            if mode == trainingmode:
                model.train()
            else:
                model.eval()
            count = 0
            for data in dataloaders[mode]:
                count += 1
                inputs, labels = data
                if gpu and cuda:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                # Forward Propagation
                output = model.forward(inputs)
                loss = criterion(output, labels)
                # Backward Propagation
                if mode == trainingmode:
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item()
                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()
            if mode == trainingmode:
                print("\nEpoch: {}/{} ".format(e+1, epochs),
                      "\nTraining Loss: {:.6f}  ".format(running_loss/count))
            else:
                print("\nValidation Loss: {:.6f}  ".format(running_loss/count),
                  "\nAccuracy: {:.6f}".format(accuracy))
            running_loss = 0
    time_elapsed = time.time() - start
    print("\nTotal Time Taken for Training: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print('Training Completed')
            
def main():
    args = parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomRotation(30),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5, 0.5, 0.5], 
                                                                     [0.25, 0.25, 0.25])])
    validataion_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.5, 0.5, 0.5], 
                                                                      [0.25, 0.25, 0.25])])
    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], 
                                                                  [0.25, 0.25, 0.25])]) 
    image_datasets = [ImageFolder(train_dir, transform=training_transforms),
                      ImageFolder(valid_dir, transform=validataion_transforms),
                      ImageFolder(test_dir, transform=testing_transforms)]
    
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=64),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64)]
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg16" or args.arch == "vgg19":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = image_datasets[0].class_to_idx
    gpu = args.gpu
    train(dataloaders, model, optimizer, criterion, epochs, gpu)
    model.class_to_idx = class_index
    save_checkpoint(model, optimizer, args, classifier)


if __name__ == "__main__":
    main()