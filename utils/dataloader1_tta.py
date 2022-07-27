import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import random
from timm.data.auto_augment import rand_augment_transform


MEAN = [0.5572, 0.3216, 0.2357] 
STD = [0.3060, 0.2145, 0.1773]


class test_dataset:
    def __init__(self, image_root, gt_root=None, testsize=384, hr=False, origin=False, rpn=False):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]    
        self.images = sorted(self.images)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((testsize, testsize)),
            transforms.Normalize(MEAN, STD)
        ])
        
        self.gts = None
        if gt_root is not None:
            self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gts = sorted(self.gts)
            self.gt_transform = transforms.ToTensor()
            
        self.size = len(self.images)
        self.index = 0
        self.origin = origin
        self.rpn = rpn
    def __len__(self):
        return self.size
    
    def load_data(self):
        image_list = []
        image_path = self.images[self.index]
        image = cv2.imread(image_path)
        
        image_tta = [image, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 
                     cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.flip(image, 0), cv2.flip(image, 1)]
        h, w, _ = image.shape 
        
        for img in image_tta:
            if self.rpn:
                image_list.append(self.transform(img).unsqueeze(0))
            else:
                tmp = []
                tmp.append(self.transform(img).unsqueeze(0))
                tmp.append(self.transform(img[:self.testsize, :self.testsize, :]).unsqueeze(0))
                tmp.append(self.transform(img[-self.testsize:, :self.testsize, :]).unsqueeze(0))
                tmp.append(self.transform(img[:self.testsize, -self.testsize:, :]).unsqueeze(0))
                tmp.append(self.transform(img[-self.testsize:, -self.testsize:, :]).unsqueeze(0))
                image_list.append(torch.cat(tmp, dim=0))
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        
        if self.gts:
            gt_path = self.gts[self.index-1]
            gt = cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)
            
            if self.origin:
                return image_list, gt, h, w, name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return image_list, gt, h, w, name
        else:
            if self.origin:
                return image_list, h, w, name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return image_list, h, w, name
