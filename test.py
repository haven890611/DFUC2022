#!/usr/bin/env python
# coding: utf-8
# %%
import torch
from torch.autograd import Variable
from datetime import datetime
from lib.TransFuse_resnet50_CSwin import TransFuse_B
from utils.dataloader import get_loader, test_dataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import shutil
import scipy
import cv2
from PIL import Image


# %%
def test(model, path, gt_path=None, save_path=None):
    model.eval()
    
    if gt_path is not None:
        image_root = '{}/images/'.format(path)
        gt_root = '{}/masks/'.format(gt_path)
        test_loader = test_dataset(image_root, gt_root, testsize=384)
    else:
        image_root = '{}/'.format(path)
        test_loader = test_dataset(image_root, testsize=384, filt=True)

        if not os.path.exists(save_path+'DFUC2022_val_release/'):
            os.makedirs(save_path+'DFUC2022_val_release/')
            
    dice_bank = []
    iou_bank = []
    acc_bank = []
    time_bank = []
    for i in range(test_loader.size):
        if gt_path is not None:
            image, gt, h, w, name= test_loader.load_data()
            image = image.cuda()
            with torch.no_grad():
                start_time = time.perf_counter()
                _, _, res = model(image)
                elapsed_time = time.perf_counter() - start_time

            res = F.interpolate(res, size=(h,w))
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = 1*(res>0.5)
            gt = gt.data.cpu().numpy().squeeze()

            smooth = 0.001
            input_flat = np.reshape(res,(-1))
            target_flat = np.reshape(gt,(-1))
            intersection = (input_flat*target_flat)
            iou =  ( intersection.sum() + smooth) / (res.sum() + gt.sum() - intersection.sum() + smooth)
            dice =  ( 2*intersection.sum() + smooth) / (res.sum() + gt.sum() + smooth)
            acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])
            
            dice_bank.append(dice)
            iou_bank.append(iou)
            acc_bank.append(acc)
            time_bank.append(elapsed_time)
            
        else:
            image, h, w, name= test_loader.load_data()
            image = image.cuda()
            with torch.no_grad():
                _, _, res = model(image)
            res = F.interpolate(res.sigmoid(), size=(h,w))
            res = res.data.cpu().numpy().squeeze()
            res = 1*(res>0.5)
            res = np.round(res*255).astype(np.uint8)
            img = Image.fromarray(res,'L')
            img.save(save_path+'DFUC2022_val_release/'+name)
            
            img = Image.open(save_path+'DFUC2022_val_release/'+name)
            img.convert('L')
            plt.imshow(img)
            plt.show()
            
    print('Dice: {:.4f}, IoU: {:.4f}'.format(np.mean(dice_bank), np.mean(iou_bank)))


# %%
test_path = "../DFUC2022_val_release"
#test_path = "../DFUC2022_train_release/val"
save_path = "/work/u6859530/DFUC/snapshots/TransFuse_B_lr_filter/"


# %%
# Create device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
model = TransFuse_B(pretrained=False).to(device)
model = torch.nn.DataParallel(model)
weight = torch.load(save_path+'best.pt')
model.load_state_dict(weight['model_state_dict'])

print("#"*20, "Start Testing", "#"*20)
#test(model, test_path, save_path=save_path)
test(model, test_path, save_path=save_path)


# %%
