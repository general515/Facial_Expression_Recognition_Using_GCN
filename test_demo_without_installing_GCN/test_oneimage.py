# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:58:32 2020

@author: Dell
"""
import torch
from PIL import Image
from net_factory import GCN_mod
#import matplotlib.pyplot as plt
from torchvision import transforms
#import time

model = GCN_mod(channel=4, lych = 10)
model.load_state_dict(torch.load('trained_RAF_10.pt'))
model.eval()

img = Image.open('t1.jpg')


"""
Ten-crop test method
"""
transform = transforms.Compose([
        transforms.Grayscale(1),        
        transforms.TenCrop(90),  
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),    
        ])

timg = transform(img)

with torch.no_grad():
    score = model(timg)

score = score.mean(dim=0)

#probability = torch.nn.functional.softmax(score, dim =0)*100

"""
recognition result
Anger:     0
Dsigust:   1
Fear:      2
Happy:     3
Netural:   4
Sadness:   5
Surprise:  6
"""
emotion = torch.argmax(score)

print("expression class is %d"%emotion)


