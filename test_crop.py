#!/usr/bin/env python
# coding: utf-8
# %%
import torch
from torch.autograd import Variable
from datetime import datetime
from lib.TransFuse_resnet50_CSwin import TransFuse_B
from utils.dataloader1 import get_loader, test_dataset
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
import csv


# %%
def test(model, path, gt_path=None, save_path=None):
    model.eval()

    if gt_path is not None:
        image_root = '{}/images/'.format(path)
        gt_root = '{}/masks/'.format(gt_path)
        test_loader = test_dataset(image_root, gt_root, testsize=384)
    else:
        image_root = '{}/'.format(path)
        test_loader = test_dataset(image_root, testsize=384)

        if not os.path.exists(save_path+'DFUC2022_val_release/'):
            os.makedirs(save_path+'DFUC2022_val_release/')
    
    dice_bank = []
    iou_bank = []
    acc_bank = []
    time_bank = []
    for i in range(test_loader.size):
        if gt_path is not None:
            image_list, gt_list, h, w, name= test_loader.load_data()
            gt = 1*(gt_list[0]>0.)
            
            image = torch.cat(image_list, dim=0)
            image = image.cuda()
            start_time = time.perf_counter()
            with torch.no_grad():
                _, _, res = model(image)
            elapsed_time = time.perf_counter() - start_time

            res0 = F.interpolate(res[0:1], size=(h,w))
            res0 = res0.sigmoid().data.cpu().numpy().squeeze()
            w = np.ones((h,w))
            #res0 = np.zeros((h,w))
            #w = np.zeros((h,w))
            
            ww = np.ones((384,384))
            w[:384,:384] = w[:384,:384] + ww
            w[-384:,:384] = w[-384:,:384] + ww
            w[:384,-384:] = w[:384,-384:] + ww
            w[-384:,-384:] = w[-384:,-384:] + ww
            
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res0[:384,:384] = res0[:384,:384] + res[1:2]
            res0[-384:,:384] = res0[-384:,:384] + res[2:3]
            res0[:384,-384:] = res0[:384,-384:] + res[3:4]
            res0[-384:,-384:] = res0[-384:,-384:] + res[4:5]

            res0 = res0/w
            res0 = (res0>0.5)
            gt = gt.data.cpu().numpy().squeeze()

            smooth = 0.001
            input_flat = np.reshape(res0,(-1))
            target_flat = np.reshape(gt,(-1))
            intersection = (input_flat*target_flat)
            iou =  ( intersection.sum() + smooth) / (res0.sum() + gt.sum() - intersection.sum() + smooth)
            dice =  ( 2*intersection.sum() + smooth) / (res0.sum() + gt.sum() + smooth)
            
            dice_bank.append(dice)
            iou_bank.append(iou)
            time_bank.append(elapsed_time)
            
        else:
            image_list, h, w, name= test_loader.load_data()
            
            image = torch.cat(image_list, dim=0)
            image = image.cuda()
            start_time = time.perf_counter()
            with torch.no_grad():
                _, _, res = model(image)
            elapsed_time = time.perf_counter() - start_time

            res0 = F.interpolate(res[0:1], size=(h,w))
            #res0 = np.zeros((h,w))
            w = np.ones((h,w))
            ww = np.ones((384,384))
            w[:384,:384] = w[:384,:384] + ww
            w[-384:,:384] = w[-384:,:384] + ww
            w[:384,-384:] = w[:384,-384:] + ww
            w[-384:,-384:] = w[-384:,-384:] + ww

            res0 = res0.sigmoid().data.cpu().numpy().squeeze()
            res = res.sigmoid().data.cpu().numpy().squeeze()

            res0[:384,:384] = res0[:384,:384] + res[1:2]
            res0[-384:,:384] = res0[-384:,:384] + res[2:3]
            res0[:384,-384:] = res0[:384,-384:] + res[3:4]
            res0[-384:,-384:] = res0[-384:,-384:] + res[4:5]

            res0 = res0/w
            res0 = (res0>0.5)
            res0 = np.round(res0*255).astype(np.uint8)
            img = Image.fromarray(res0,'L')
            img.save(save_path+'DFUC2022_val_release/'+name)
            img = Image.open(save_path+'DFUC2022_val_release/'+name)
            img.convert('L')
            print(name)
            plt.imshow(img)
            plt.show()
            
    print('Dice: {:.4f}, IoU: {:.4f}, fps: {:.4f}'.format(np.mean(dice_bank), np.mean(iou_bank), 1./np.mean(time_bank)))


# %%
def test5(model, path, gt_path=None, save_path=None):
    model.eval()

    if gt_path is not None:
        image_root = '{}/images/'.format(path)
        gt_root = '{}/masks/'.format(gt_path)
        test_loader = test_dataset(image_root, gt_root, testsize=384, origin=True)
    else:
        image_root = '{}/'.format(path)
        test_loader = test_dataset(image_root, testsize=384, origin=True)

        if not os.path.exists(save_path+'DFUC2022_val_release/'):
            os.makedirs(save_path+'DFUC2022_val_release/')
    
    dice_bank = []
    iou_bank = []
    acc_bank = []
    time_bank = []
    for i in range(test_loader.size):
        if gt_path is not None:
            image_list, gt_list, h, w, name, origin= test_loader.load_data()
            gt = 1*(gt_list[0]>0.)
            
            image = torch.cat(image_list, dim=0)
            image = image.cuda()
            start_time = time.perf_counter()
            with torch.no_grad():
                _, _, res = model(image)
            elapsed_time = time.perf_counter() - start_time
            res0 = F.interpolate(res[0:1], size=(h,w))
            w0 = 2*torch.abs(F.avg_pool2d(res0.sigmoid(), kernel_size=7, stride=1, padding=3)-0.5)
            res0 = res0.squeeze()
            w0 = w0.squeeze()
            
            ww = torch.ones(384,384).to(device)
            w1 = torch.zeros(h,w).to(device)
            w1[:384,:384] = w1[:384,:384] + ww
            w1[-384:,:384] = w1[-384:,:384] + ww
            w1[:384,-384:] = w1[:384,-384:] + ww
            w1[-384:,-384:] = w1[-384:,-384:] + ww
            
            res1 = torch.zeros(h,w).to(device)
            res1[:384,:384] = res1[:384,:384] + res[1,0]
            res1[-384:,:384] = res1[-384:,:384] + res[2,0]
            res1[:384,-384:] = res1[:384,-384:] + res[3,0]
            res1[-384:,-384:] = res1[-384:,-384:] + res[4,0]

            res_final = (res1+res0*w0)/(w0+w1)
            res_final = res_final.sigmoid().data.cpu().numpy().squeeze()
            res_final = (res_final>0.5)
            gt = gt.data.cpu().numpy().squeeze()

            smooth = 0.001
            input_flat = np.reshape(res_final,(-1))
            target_flat = np.reshape(gt,(-1))
            intersection = (input_flat*target_flat)
            iou =  ( intersection.sum() + smooth) / (res_final.sum() + gt.sum() - intersection.sum() + smooth)
            dice =  ( 2*intersection.sum() + smooth) / (res_final.sum() + gt.sum() + smooth)
            
            dice_bank.append(dice)
            iou_bank.append(iou)
            time_bank.append(elapsed_time)
            
        else:
            image_list, h, w, name, origin= test_loader.load_data()
            print(name)
            image = torch.cat(image_list, dim=0)
            image = image.cuda()
            start_time = time.perf_counter()
            with torch.no_grad():
                _, _, res = model(image)
            elapsed_time = time.perf_counter() - start_time
            res0 = F.interpolate(res[0:1], size=(h,w))
            w0 = 2*torch.abs(F.avg_pool2d(res0.sigmoid(), kernel_size=7, stride=1, padding=3)-0.5)
            res0 = res0.squeeze()
            w0 = w0.squeeze()
            
            ww = torch.ones(384,384).to(device)
            w1 = torch.zeros(h,w).to(device)
            w1[:384,:384] = w1[:384,:384] + ww
            w1[-384:,:384] = w1[-384:,:384] + ww
            w1[:384,-384:] = w1[:384,-384:] + ww
            w1[-384:,-384:] = w1[-384:,-384:] + ww
            
            res1 = torch.zeros(h,w).to(device)
            res1[:384,:384] = res1[:384,:384] + res[1,0]
            res1[-384:,:384] = res1[-384:,:384] + res[2,0]
            res1[:384,-384:] = res1[:384,-384:] + res[3,0]
            res1[-384:,-384:] = res1[-384:,-384:] + res[4,0]

            res_final = (res1+res0*w0)/(w0+w1)
            res_final = res_final.sigmoid().data.cpu().numpy().squeeze()
            res1 = (res1/w1).sigmoid().data.cpu().numpy().squeeze()
            res0 = res0.sigmoid().data.cpu().numpy().squeeze()
            overlap = np.zeros((h,w,3))
            overlap[:,:,0] = res0
            overlap[:,:,1] = res1
            #overlap[:,:,2] = res_final
            
            f, axarr = plt.subplots(1,2,figsize=(10,10))
            axarr[0].imshow(overlap)
            axarr[1].imshow(origin)
            plt.show()
            #res_final = (res_final>0.5)
            #res_final = np.round(res_final*255).astype(np.uint8)
            #img = Image.fromarray(res_final,'L')
            #img.save(save_path+'DFUC2022_val_release/'+name)
            #
            #img = Image.open(save_path+'DFUC2022_val_release/'+name)
            #img.convert('L')
            #plt.imshow(img)
            #plt.show()
            
    print('Dice: {:.4f}, IoU: {:.4f}, fps: {:.4f}'.format(np.mean(dice_bank), np.mean(iou_bank), 1./np.mean(time_bank)))


# %%
def test6(model, path, gt_path=None, save_path=None):
    model.eval()

    if gt_path is not None:
        image_root = '{}/images/'.format(path)
        gt_root = '{}/masks/'.format(gt_path)
        test_loader = test_dataset(image_root, gt_root, testsize=384, origin=True)
    else:
        image_root = '{}/'.format(path)
        test_loader = test_dataset(image_root, testsize=384, origin=True)

        if not os.path.exists(save_path+'DFUC2022_val_release/'):
            os.makedirs(save_path+'DFUC2022_val_release/')
    
    dice_bank = []
    iou_bank = []
    acc_bank = []
    time_bank = []
    for i in range(test_loader.size):
        if gt_path is not None:
            image_list, gt_list, h, w, name, origin= test_loader.load_data()
            gt = 1*(gt_list[0]>0.)
            
            image = torch.cat(image_list, dim=0)
            image = image.cuda()
            start_time = time.perf_counter()
            with torch.no_grad():
                _, _, res = model(image)
            elapsed_time = time.perf_counter() - start_time
            res0 = F.interpolate(res[0:1], size=(h,w))
            w0 = 2*torch.abs(F.avg_pool2d(res0.sigmoid(), kernel_size=7, stride=1, padding=3)-0.5)
            res0 = res0.squeeze()
            w0 = w0.squeeze()
            
            ww = torch.ones(384,384).to(device)
            w1 = torch.zeros(h,w).to(device)
            w1[:384,:384] = w1[:384,:384] + ww
            w1[-384:,:384] = w1[-384:,:384] + ww
            w1[:384,-384:] = w1[:384,-384:] + ww
            w1[-384:,-384:] = w1[-384:,-384:] + ww
            
            res1 = torch.zeros(h,w).to(device)
            res1[:384,:384] = res1[:384,:384] + res[1,0]
            res1[-384:,:384] = res1[-384:,:384] + res[2,0]
            res1[:384,-384:] = res1[:384,-384:] + res[3,0]
            res1[-384:,-384:] = res1[-384:,-384:] + res[4,0]

            res_final = (res1+res0*w0)/(w0+w1)
            res_final = res_final.sigmoid().data.cpu().numpy().squeeze()
            res_final = (res_final>0.5)
            gt = gt.data.cpu().numpy().squeeze()

            smooth = 0.001
            input_flat = np.reshape(res_final,(-1))
            target_flat = np.reshape(gt,(-1))
            intersection = (input_flat*target_flat)
            iou =  ( intersection.sum() + smooth) / (res_final.sum() + gt.sum() - intersection.sum() + smooth)
            dice =  ( 2*intersection.sum() + smooth) / (res_final.sum() + gt.sum() + smooth)
            
            dice_bank.append(dice)
            iou_bank.append(iou)
            time_bank.append(elapsed_time)
            
        else:
            image_list, h, w, name, origin= test_loader.load_data()
            print(name)
            image = torch.cat(image_list, dim=0)
            image = image.cuda()
            start_time = time.perf_counter()
            with torch.no_grad():
                _, _, res = model(image)
            elapsed_time = time.perf_counter() - start_time
            res0 = F.interpolate(res[0:1], size=(h,w))
            w0 = 2*torch.abs(F.avg_pool2d(res0.sigmoid(), kernel_size=7, stride=1, padding=3)-0.5)
            w00 = torch.abs(F.avg_pool2d(res0.sigmoid(), kernel_size=5, stride=1, padding=2)-0.3)
            res0 = res0.squeeze()
            w0 = w0.squeeze()
            
            ww = torch.ones(384,384).to(device)
            w1 = torch.zeros(h,w).to(device)
            w1[:384,:384] = w1[:384,:384] + ww
            w1[-384:,:384] = w1[-384:,:384] + ww
            w1[:384,-384:] = w1[:384,-384:] + ww
            w1[-384:,-384:] = w1[-384:,-384:] + ww
            
            res1 = torch.zeros(h,w).to(device)
            res1[:384,:384] = res1[:384,:384] + res[1,0]
            res1[-384:,:384] = res1[-384:,:384] + res[2,0]
            res1[:384,-384:] = res1[:384,-384:] + res[3,0]
            res1[-384:,-384:] = res1[-384:,-384:] + res[4,0]

            res_final = (res1+res0*w0)/(w0+w1)
            res_final2 = (res1*(1-w0)+res0*w0)/(w0+w1*(1-w0))
            res_final3 = (res1*(1-w00)+res0*w0)/(w0+w1*(1-w00))
            
            res_final = res_final.sigmoid().data.cpu().numpy().squeeze()
            res_final2 = res_final2.sigmoid().data.cpu().numpy().squeeze()
            res_final3 = res_final3.sigmoid().data.cpu().numpy().squeeze()
            
            res_final = (res_final>0.5)
            res_final3 = (res_final3>0.5)
            overlap = np.zeros((h,w,3))
            overlap[:,:,0] = res_final
            overlap[:,:,1] = res_final3      
            f, axarr = plt.subplots(1,2,figsize=(20,10))
            axarr[0].imshow(overlap)
            axarr[1].imshow(origin)
            plt.show()
            
            res_final3 = np.round(res_final3*255).astype(np.uint8)
            img = Image.fromarray(res_final3,'L')
            img.save(save_path+'DFUC2022_val_release/'+name)
            #
            #img = Image.open(save_path+'DFUC2022_val_release/'+name)
            #img.convert('L')
            #plt.imshow(img)
            #plt.show()
            
    print('Dice: {:.4f}, IoU: {:.4f}, fps: {:.4f}'.format(np.mean(dice_bank), np.mean(iou_bank), 1./np.mean(time_bank)))


# %%
def test5_rpn(model, model1, path, gt_path=None, save_path=None, save_path1=None):
    model.eval()
    model1.eval()
    if gt_path is not None:
        image_root = '{}/images/'.format(path)
        gt_root = '{}/masks/'.format(gt_path)
        test_loader = test_dataset(image_root, gt_root, testsize=384, origin=True)
        image_root_rpn = '../RPN/RP/val/images/'
        gt_root_rpn = '../RPN/RP/val/masks/'
        test_loader_rpn = test_dataset(image_root_rpn, gt_root_rpn, testsize=384)
        with open('../RPN/val_top_lefts.csv', newline='') as f:
            reader = csv.reader(f)
            left_top = list(reader)
        
    else:
        image_root = '{}/'.format(path)
        test_loader = test_dataset(image_root, testsize=384, origin=True)
        image_root_rpn = '../RPN/RP/test/'
        test_loader_rpn = test_dataset(image_root_rpn, testsize=384)
        with open('../RPN/test_top_lefts.csv', newline='') as f:
            reader = csv.reader(f)
            left_top = list(reader)
        if not os.path.exists(save_path1+'DFUC2022_val_release/'):
            os.makedirs(save_path1+'DFUC2022_val_release/')
    cnt = 0
    dice_bank = []
    iou_bank = []
    for i in range(test_loader.size):
        if gt_path is not None:
            image_list, gt_list, h, w, name, origin= test_loader.load_data()
            gt = 1*(gt_list[0]>0.)
            image = torch.cat(image_list, dim=0)
            image = image.cuda()
            with torch.no_grad():
                _, _, res = model(image)
            
            res0 = F.interpolate(res[0:1], size=(h,w))
            w0 = 2*torch.abs(F.avg_pool2d(res0.sigmoid(), kernel_size=15, stride=1, padding=7)-0.5)
            res0 = res0.squeeze()
            w0 = w0.squeeze()
            ww = torch.ones(384,384).to(device)
            w1 = torch.zeros(h,w).to(device)
            w1[:384,:384] = w1[:384,:384] + ww
            w1[-384:,:384] = w1[-384:,:384] + ww
            w1[:384,-384:] = w1[:384,-384:] + ww
            w1[-384:,-384:] = w1[-384:,-384:] + ww
            res1 = torch.zeros(h,w).to(device)
            res1[:384,:384] = res1[:384,:384] + res[1,0]
            res1[-384:,:384] = res1[-384:,:384] + res[2,0]
            res1[:384,-384:] = res1[:384,-384:] + res[3,0]
            res1[-384:,-384:] = res1[-384:,-384:] + res[4,0]
            
            ww_rpn = torch.zeros(h,w).to(device)
            res_rpn = torch.zeros(h,w).to(device)
            if name.split('.')[0] == (left_top[cnt][0]).split('.')[0]:
                image_list_rpn, gt_list_rpn, h_rpn, w_rpn, name_rpn= test_loader_rpn.load_data()
                image_rpn = image_list_rpn[0].cuda()
                top = int(left_top[cnt][1])
                down = top+384
                left = int(left_top[cnt][2])
                right = left+384
                cnt = cnt+1
                with torch.no_grad():
                    _, _, res_rpn = model1(image_rpn)
                res_rpn = F.pad(res_rpn[0,0], (left, w-right, top, h-down), "constant", -20.)
                ww_rpn[top:down, left:right] = ww_rpn[top:down, left:right] + ww
            
            res_final = (torch.ones(h,w).to(device)-ww_rpn)*((res1+res0*w0)/(w0+w1)) + ww_rpn*res_rpn
            f, axarr = plt.subplots(1,2,figsize=(10,5))
            axarr[0].imshow(res_rpn.sigmoid().data.cpu().numpy().squeeze())
            axarr[1].imshow(res_final.sigmoid().data.cpu().numpy().squeeze())
            plt.show()
            
            res_final = res_final.sigmoid().data.cpu().numpy().squeeze()
            res_final = (res_final>0.5)
            gt = gt.data.cpu().numpy().squeeze()

            smooth = 0.001
            input_flat = np.reshape(res_final,(-1))
            target_flat = np.reshape(gt,(-1))
            intersection = (input_flat*target_flat)
            iou =  ( intersection.sum() + smooth) / (res_final.sum() + gt.sum() - intersection.sum() + smooth)
            dice =  ( 2*intersection.sum() + smooth) / (res_final.sum() + gt.sum() + smooth)
            
            dice_bank.append(dice)
            iou_bank.append(iou)
            
        else:
            image_list, h, w, name, origin= test_loader.load_data()
            print(name)
            image = torch.cat(image_list, dim=0)
            image = image.cuda()
            with torch.no_grad():
                _, _, res = model(image)
            
            res0 = F.interpolate(res[0:1], size=(h,w))
            w0 = 2*torch.abs(F.avg_pool2d(res0.sigmoid(), kernel_size=15, stride=1, padding=7)-0.5)
            res0 = res0.squeeze()
            w0 = w0.squeeze()
            ww = torch.ones(384,384).to(device)
            w1 = torch.zeros(h,w).to(device)
            w1[:384,:384] = w1[:384,:384] + ww
            w1[-384:,:384] = w1[-384:,:384] + ww
            w1[:384,-384:] = w1[:384,-384:] + ww
            w1[-384:,-384:] = w1[-384:,-384:] + ww
            res1 = torch.zeros(h,w).to(device)
            res1[:384,:384] = res1[:384,:384] + res[1,0]
            res1[-384:,:384] = res1[-384:,:384] + res[2,0]
            res1[:384,-384:] = res1[:384,-384:] + res[3,0]
            res1[-384:,-384:] = res1[-384:,-384:] + res[4,0]
            
            ww_rpn = torch.zeros(h,w).to(device)
            res_rpn = torch.zeros(h,w).to(device)
            if name.split('.')[0] == (left_top[cnt][0]).split('.')[0]:
                image_list_rpn, h_rpn, w_rpn, name_rpn= test_loader_rpn.load_data()
                image_rpn = image_list_rpn[0].cuda()
                top = int(left_top[cnt][1])
                down = top+384
                left = int(left_top[cnt][2])
                right = left+384
                cnt = cnt+1
                with torch.no_grad():
                    _, _, res_rpn = model1(image_rpn)
                res_rpn = F.pad(res_rpn[0,0], (left, w-right, top, h-down), "constant", -20.)
                ww_rpn[top:down, left:right] = ww_rpn[top:down, left:right] + ww*0.5
                #ww_rpn[top:down, left:right] = ww_rpn[top:down, left:right] + ww
            
            res_no_rpn = (res1+res0*w0)/(w0+w1)
            res_no_rpn = res_no_rpn.sigmoid().data.cpu().numpy().squeeze()
            res_no_rpn = (res_no_rpn>0.5)
            res_no_rpn = np.round(res_no_rpn*255).astype(np.uint8)
            
            res_final = (torch.ones(h,w).to(device)-ww_rpn)*((res1+res0*w0)/(w0+w1)) + ww_rpn*res_rpn
            #res_final = (res1+res0*w0+ww_rpn*res_rpn)/(w0+w1+ww_rpn)
            res_final = res_final.sigmoid().data.cpu().numpy().squeeze()
            res_final = (res_final>0.5)
            res_final = np.round(res_final*255).astype(np.uint8)
            
            overlap = np.zeros((h,w,3))
            overlap[:,:,0] = res_no_rpn
            overlap[:,:,1] = res_final
            
            img = Image.fromarray(res_final,'L')
            img.save(save_path1+'DFUC2022_val_release/'+name)
            img = Image.open(save_path1+'DFUC2022_val_release/'+name)
            img.convert('L')
            
            f, axarr = plt.subplots(1,3,figsize=(15,10))
            axarr[0].imshow(overlap)
            axarr[1].imshow(img)
            axarr[2].imshow(origin)
            plt.show()
            
    print('Dice: {:.4f}, IoU: {:.4f}'.format(np.mean(dice_bank), np.mean(iou_bank)))


# %%
test_path = "../DFUC2022_val_release"
#test_path = "../DFUC2022_train_release/val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
save_path = "/work/u6859530/DFUC/snapshots/crop_all_iou_focal_k7_test2/"
model = TransFuse_B(pretrained=False).to(device)
model = torch.nn.DataParallel(model)
weight = torch.load(save_path+'best.pt')
model.load_state_dict(weight['model_state_dict'])

#save_path1 = "/work/u6859530/DFUC/snapshots/TransFuse_B_rpn_k7_jitter0.0/"
#model1 = TransFuse_B(pretrained=False).to(device)
#model1 = torch.nn.DataParallel(model1)
#weight1 = torch.load(save_path1+'best.pt')
#model1.load_state_dict(weight1['model_state_dict'])


# %%
print("#"*20, "Start Testing", "#"*20)

#test5(model, test_path, gt_path=test_path, save_path=save_path)
test5(model, test_path, save_path=save_path)
#test5_rpn(model, model1, test_path, gt_path=test_path , save_path=save_path)
#test5_rpn(model, model1, test_path, save_path=save_path, save_path1=save_path1)


# %%
