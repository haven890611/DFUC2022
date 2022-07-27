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


class KvasirDataset(data.Dataset):
    """
    dataloader for skin lesion segmentation tasks
    """
    def __init__(self, image_root, gt_root, img_size, group=5, jitter=0.5):

        self.group = group
        self.img_size = img_size
        
        self.images_all = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_all.sort()
        self.images = []
        self.gts_all = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts_all.sort()
        self.gts = []
        
        self.size = len(self.images_all)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(MEAN, STD)
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
        ])

        self.transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
            A.ColorJitter(jitter, jitter, jitter),
            A.HorizontalFlip(),
            A.VerticalFlip()
        ])
        
    def split(self):
        length = len(self.images_all)
        c = list(zip(self.images_all, self.gts_all))
        random.shuffle(c)
        self.images_all, self.gts_all = zip(*c)
        self.images = self.images_all[length//self.group:]
        self.gts = self.gts_all[length//self.group:]
        
        return self.images_all[:length//self.group], self.gts_all[:length//self.group]
        
    def __getitem__(self, index):
        image_path = self.images[index]
        gt_path = self.gts[index]
        image = cv2.imread(image_path)
        gt = cv2.imread(gt_path)
        gt = gt/255.0

        index = random.randint(1,4)
        if index == 4:
            image1 = image[:self.img_size,:self.img_size,:]
            gt1 = gt[:self.img_size,:self.img_size,:]
        elif index == 3:
            image1 = image[-self.img_size:,:self.img_size,:]
            gt1 = gt[-self.img_size:,:self.img_size,:]
        elif index == 2:
            image1 = image[:self.img_size,-self.img_size:,:]
            gt1 = gt[:self.img_size,-self.img_size:,:]
        else:
            image1 = image[-self.img_size:,-self.img_size:,:]
            gt1 = gt[-self.img_size:,-self.img_size:,:]
        
        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = self.gt_transform(transformed['mask'])
        
        transformed1 = self.transform(image=image1, mask=gt1)
        image1 = self.img_transform(transformed1['image'])
        gt1 = self.gt_transform(transformed1['mask'])
        
        return image, gt, image1, gt1

    def __len__(self):
        return self.size*(self.group-1)//self.group


class test_dataset:
    def __init__(self, images, gts=None, testsize=384, hr=False, origin=False):
        self.testsize = testsize
        self.images = images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((testsize, testsize)),
            transforms.Normalize(MEAN, STD)
        ])
        
        self.gts = gts
        self.gt_transform = transforms.ToTensor()
            
        self.size = len(self.images)
        self.index = 0
        self.origin = origin
        
    def __len__(self):
        return self.size
    
        
    def load_data(self):
        image_list = []
        image_path = self.images[self.index]
        image = cv2.imread(image_path)
        h, w, _ = image.shape 
        image_list.append(self.transform(image).unsqueeze(0))
        image_list.append(self.transform(image[:self.testsize, :self.testsize, :]).unsqueeze(0))
        image_list.append(self.transform(image[-self.testsize:, :self.testsize, :]).unsqueeze(0))
        image_list.append(self.transform(image[:self.testsize, -self.testsize:, :]).unsqueeze(0))
        image_list.append(self.transform(image[-self.testsize:, -self.testsize:, :]).unsqueeze(0))
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        
        if self.gts:
            gt_list = []
            gt_path = self.gts[self.index]
            gt = cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)
            gt_list.append(self.gt_transform(gt).unsqueeze(0))
            gt_list.append(self.gt_transform(gt[:self.testsize,:self.testsize]).unsqueeze(0))
            gt_list.append(self.gt_transform(gt[-self.testsize:,:self.testsize]).unsqueeze(0))
            gt_list.append(self.gt_transform(gt[:self.testsize,-self.testsize:]).unsqueeze(0))
            gt_list.append(self.gt_transform(gt[-self.testsize:,-self.testsize:]).unsqueeze(0))
            
            if self.origin:
                return image_list, gt_list, h, w, name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return image_list, gt_list, h, w, name
        else:
            if self.origin:
                return image_list, h, w, name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return image_list, h, w, name
            
        self.index += 1


def get_loader(image_root, gt_root, batchsize, img_size=384, shuffle=True, num_workers=4, pin_memory=True, group=5, jitter=0.5):
    trainset = KvasirDataset(image_root, gt_root, img_size=img_size, group=group, jitter=jitter) 
    
    return trainset
