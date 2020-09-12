from __future__ import division

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torchvision import datasets, transforms
from GCN_net import GCN7

from utils import (AverageMeter, accuracy, clip_gradient, get_parameters_size,
                   save_checkpoint)



parser = argparse.ArgumentParser(description='PyTorch GCN MNIST Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--b', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--gpu', default=1, type=int,
                    metavar='N', help='CPU: 0; GPU 1 (default: 1)')
parser.add_argument('--dset', default=2, type=int,
                    metavar='N', help='Choice dataset 1 = RAF, 2 = FER2013 (default: 1)')
args = parser.parse_args()

use_cuda = (args.gpu >= 0) and torch.cuda.is_available()

transform = {
    'train': transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(100),
        transforms.RandomCrop(90),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.ToTensor(),          
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(100),
        transforms.TenCrop(90),  
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),    
    ]),

    'test': transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(100),
        transforms.TenCrop(90),  
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),    
    ]),
}

if args.dset == 1:
    data_dir = 'D:\\Datasets\\raf\\'    
    image_datasets = {x: datasets.ImageFolder(
                    data_dir + x,
                    transform[x])
                    for x in ['train', 'test']}
    train_loader = Data.DataLoader(dataset = image_datasets['train'], batch_size = args.b, shuffle=True)
    test_loader = Data.DataLoader(dataset = image_datasets['test'], batch_size = 32, shuffle=True)
    
elif args.dset == 2:
    data_dir = 'D:\\Datasets\\fer2013\\datasets\\'
    #data_dir = 'D:\\Datasets\\fer2013_plus\\'    
    image_datasets = {x: datasets.ImageFolder(
                    data_dir + x,
                    transform[x])
                    for x in ['train', 'test', 'val']}
    train_loader = Data.DataLoader(dataset = image_datasets['train'], batch_size = args.b, shuffle=True)
    test_loader = Data.DataLoader(dataset = image_datasets['test'], batch_size = 32, shuffle=True)
    val_loader = Data.DataLoader(dataset = image_datasets['val'], batch_size = 32, shuffle=True)     
else:
    print("Dataset error")
    exit(1)

# Load model

model = GCN7(channel=4, lych=10)

print(model)

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=3e-04)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if use_cuda else "cpu")
model = model.to(device)

criterion = criterion.to(device)
# Calculate the total parameters of the model
print('Model size: {:0.2f} million float parameters'.format(get_parameters_size(model)/1e6))

if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))

def train(epoch):
    model.train()
    global iteration
    st = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() 
        output = model(data)
        loss = criterion(output, target)
        prec1, = accuracy(output, target)
        loss.backward()  
        optimizer.step()

        
        if batch_idx % args.print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), prec1.item()))           
    
    scheduler.step()
    
def test(epoch):
    model.eval()
    test_loss = AverageMeter()
    acc = AverageMeter()
    with torch.no_grad():
        for data, target in test_loader:
            bs, ncrops, c,h,w = data.size()
            data = data.view(-1,c,h,w)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.view(bs,ncrops,-1).mean(1)

            test_loss.update(F.cross_entropy(output, target, reduction='mean').item(), target.size(0))
            prec1, = accuracy(output, target) # test precison in one batch
            acc.update(prec1.item(), target.size(0))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss.avg, acc.avg))
    return acc.avg


def val(epoch):
    model.eval()
    test_loss = AverageMeter()
    acc = AverageMeter()
    with torch.no_grad():
        for data, target in val_loader:
            bs, ncrops, c,h,w = data.size()
            data = data.view(-1,c,h,w)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.view(bs,ncrops,-1).mean(1)

            test_loss.update(F.cross_entropy(output, target, reduction='mean').item(), target.size(0))
            prec1, = accuracy(output, target) # test precison in one batch
            acc.update(prec1.item(), target.size(0))

    return acc.avg




best_t = 0
best_v = 0
if args.dset == 1:    
    for epoch in range(args.start_epoch, args.epochs):
        print('------------------------------------------------------------------------')
        st = time.time()
        train(epoch+1)       
        
        prec_t = test(epoch+1)      
        is_best_t = prec_t > best_t
        best_t = max(prec_t, best_t)
        if is_best_t:
            torch.save({ 
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_t,
                    'optimizer' : optimizer.state_dict(),
            }, 'best_test_para.pth.tar')      
        print('Best Test Precision@top1:{:.2f}%'.format(best_t))        
        epoch_time = time.time() - st    
        print('Epoch time:{:0.2f}s'.format(epoch_time))
    
else:
    for epoch in range(args.start_epoch, args.epochs):
        print('------------------------------------------------------------------------')
        st = time.time()
        train(epoch+1)
        
        prec_t = test(epoch+1)
        torch.save(model.state_dict(),'final.pt')
    
        is_best_t = prec_t > best_t
        best_t = max(prec_t, best_t)
        if is_best_t:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_t,
                'optimizer' : optimizer.state_dict(),
        }, 'best_test_para.pth.tar')            
        print('Best Test Precision@top1:{:.2f}%'.format(best_t))

        prec_v = val(epoch+1)
        is_best_v = prec_v > best_v
        best_v = max(prec_v, best_v)        
        if is_best_v:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_v,
                'optimizer' : optimizer.state_dict(),
        }, 'best_val_para.pth.tar')            
        print('Best Val Precision@top1:{:.2f}%'.format(best_v))
        
        epoch_time = time.time() - st    
        print('\nEpoch time:{:0.2f}s'.format(epoch_time))

print('\nFinished!')
