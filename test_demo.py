from __future__ import division

import argparse
import time
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import datasets, transforms
from GCN_net import GCN7
from utils import (AverageMeter, accuracy)


parser = argparse.ArgumentParser(description='PyTorch GCN MNIST Training')

parser.add_argument('--gpu', default=1, type=int,
                    metavar='N', help='CPU :0, GPU: 1 (default: 1)')
args = parser.parse_args()



use_cuda = (args.gpu > 0) and torch.cuda.is_available()


transform =  transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(100),
        transforms.TenCrop(90),  
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),    
    ])

"""
Please revise the path for dataset
"""
data_dir = 'D:\\Datasets\\raf\\'
#data_dir = 'D:\\Datasets\\fer2013\\datasets\\'
#data_dir = 'D:\\Datasets\\fer2013_plus\\'

image_datasets = datasets.ImageFolder(data_dir + 'test', transform)
test_loader = Data.DataLoader(dataset = image_datasets, batch_size = 32)

"""
GCN10 == GCN7(channel=4, lych = 10)
GCN40 == GCN7(channel=4, lych = 40)
"""
model = GCN7(channel=4, lych = 10)
model.load_state_dict(torch.load('./trained_weights/GCN10_RAF.pt'))

print(model)

device = torch.device("cuda" if use_cuda else "cpu")
model = model.to(device)

#print('Model size: {:0.2f} million float parameters'.format(get_parameters_size(model)/1e6))

def test():
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
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss.avg, acc.avg))
    return acc.avg

st = time.time()
prediction= test()
epoch_time = time.time() - st
print('Accuracy is: %.2f%%' %prediction)
print('Test time:{:0.2f}s'.format(epoch_time))


print('\nFinished!')
