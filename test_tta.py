#!/usr/bin/env python
# coding: utf-8
# %%
import torch
from torch.autograd import Variable
from datetime import datetime
from lib.TransFuse_resnet50_CSwin import TransFuse_B
from utils.dataloader1_tta import test_dataset
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import shutil
import scipy
import cv2
import csv
from PIL import Image


# %%
def test(model, path, gt_path=None, save_path=None, sota_path=None, ref_path=None):
    model.eval()

    image_root = '{}/'.format(path)
    test_loader = test_dataset(image_root, testsize=384, origin=True)

    if not os.path.exists(save_path+'DFUC2022_val_release/'):
        os.makedirs(save_path+'DFUC2022_val_release/')
    
    for i in range(test_loader.size):
        image_list, h, w, name, origin= test_loader.load_data()
        image = torch.cat(image_list, dim=0)
        image = image.cuda()
        print(name)
        
        f, axarr = plt.subplots(1,3,figsize=(24,10))
        final = np.zeros((h,w))
        overlap = np.zeros((h,w,3))
        with torch.no_grad():
            _, _, res = model(image)

        for idx in range(len(image_list)):
            cur = idx*5
            
            if idx==1 or idx==2 :
                res0 = F.interpolate(res[cur:cur+1], size=(w,h))
                w1 = torch.zeros(w,h).to(device)
                res1 = torch.zeros(w,h).to(device)
            else:
                res0 = F.interpolate(res[cur:cur+1], size=(h,w))
                w1 = torch.zeros(h,w).to(device)
                res1 = torch.zeros(h,w).to(device)
                
            w0 = 2*torch.abs(F.avg_pool2d(res0.sigmoid(), kernel_size=7, stride=1, padding=3)-0.5)
            #w0 = 2*torch.abs(F.avg_pool2d(res0.sigmoid(), kernel_size=7, stride=1, padding=3)-0.7)
            res0 = res0.squeeze()
            w0 = w0.squeeze()
            
            ww = torch.ones(384,384).to(device)
            w1[:384,:384] = w1[:384,:384] + ww
            w1[-384:,:384] = w1[-384:,:384] + ww
            w1[:384,-384:] = w1[:384,-384:] + ww
            w1[-384:,-384:] = w1[-384:,-384:] + ww
            
            
            res1[:384,:384] = res1[:384,:384] + res[cur+1,0]
            res1[-384:,:384] = res1[-384:,:384] + res[cur+2,0]
            res1[:384,-384:] = res1[:384,-384:] + res[cur+3,0]
            res1[-384:,-384:] = res1[-384:,-384:] + res[cur+4,0]
            
            res_final = (res1+res0*w0)/(w0+w1)
            #res_final = (res1*(1-w0)+res0*w0)/(w1*(1-w0)+w0)
            res_final = res_final.sigmoid().data.cpu().numpy().squeeze()
            res_final = (res_final>0.5)*1
            
            if idx==1:
                res_final = cv2.rotate(res_final, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif idx==2:
                res_final = cv2.rotate(res_final, cv2.ROTATE_90_CLOCKWISE)
            elif idx==3:
                res_final = cv2.flip(res_final, 0)
            elif idx==4:
                res_final = cv2.flip(res_final, 1)
            final += res_final
        
        final = (final>2)*255
        cv2.imwrite(save_path+'DFUC2022_val_release/'+name, final)
        
        sota = cv2.imread(sota_path+name, cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(ref_path+name, cv2.IMREAD_GRAYSCALE)
        overlap[:,:,0] = sota
        overlap[:,:,1] = final
        axarr[0].imshow(overlap)
        axarr[1].imshow(origin)
        axarr[2].imshow(ref)
        
        plt.show()


# %%
def test_rpn(model, path, gt_path=None, save_path=None, sota_path=None, ref_path=None):
    model.eval()

    image_root = '{}/'.format(path)
    test_loader = test_dataset(image_root, testsize=384, origin=True)
    image_root_rpn = '../RPN/RP/test/'
    test_loader_rpn = test_dataset(image_root_rpn, testsize=384, rpn=True)
    with open('../RPN/test_top_lefts.csv', newline='') as f:
        reader = csv.reader(f)
        left_top = list(reader)
        
    if not os.path.exists(save_path+'DFUC2022_val_release/'):
        os.makedirs(save_path+'DFUC2022_val_release/')
    
    cnt=0
    for i in range(test_loader.size):
        image_list, h, w, name, origin= test_loader.load_data()
        image = torch.cat(image_list, dim=0)
        image = image.cuda()
        print(name)
        
        f, axarr = plt.subplots(1,3,figsize=(24,10))
        final = np.zeros((h,w))
        overlap = np.zeros((h,w,3))
        ww = torch.ones(384,384).to(device)

        with torch.no_grad():
            _, _, res = model(image)
            
        rpn_list=[]
        ww_rpn_list=[]
        if name.split('.')[0] == (left_top[cnt][0]).split('.')[0]:
            ww_rpn = torch.zeros(h,w).to(device)
            image_list_rpn, _, _, _= test_loader_rpn.load_data()
            image_rpn = torch.cat(image_list_rpn, dim=0)
            image_rpn = image_rpn.cuda()
            top = int(left_top[cnt][1])
            down = top+384
            left = int(left_top[cnt][2])
            right = left+384
            cnt = cnt+1
            with torch.no_grad():
                _, _, res_ = model(image_rpn)
            ww_rpn[top:down, left:right] = ww_rpn[top:down, left:right] + ww
            
            for idx in range(len(image_list_rpn)):
                res_final_rpn = res_[idx:idx+1]
                res_final_rpn = res_final_rpn.data.cpu().numpy().squeeze()
                if idx==1:
                    res_final_rpn = cv2.rotate(res_final_rpn, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif idx==2:
                    res_final_rpn = cv2.rotate(res_final_rpn, cv2.ROTATE_90_CLOCKWISE)
                elif idx==3:
                    res_final_rpn = cv2.flip(res_final_rpn, 0)
                elif idx==4:
                    res_final_rpn = cv2.flip(res_final_rpn, 1)
                res_final_rpn = np.pad(res_final_rpn, ((top, h-down), (left, w-right)), 'constant', constant_values=-20)
                
                ww_rpn_tmp = ww_rpn.data.cpu().numpy().squeeze()
                if idx==1:
                    res_final_rpn = cv2.rotate(res_final_rpn, cv2.ROTATE_90_CLOCKWISE)
                    ww_rpn_tmp = cv2.rotate(ww_rpn_tmp, cv2.ROTATE_90_CLOCKWISE)
                elif idx==2:
                    res_final_rpn = cv2.rotate(res_final_rpn, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    ww_rpn_tmp = cv2.rotate(ww_rpn_tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif idx==3:
                    res_final_rpn = cv2.flip(res_final_rpn, 0)
                    ww_rpn_tmp = cv2.flip(ww_rpn_tmp, 0)
                elif idx==4:
                    res_final_rpn = cv2.flip(res_final_rpn, 1)
                    ww_rpn_tmp = cv2.flip(ww_rpn_tmp, 1)
                rpn_list.append(transforms.ToTensor()(res_final_rpn).to(device))
                ww_rpn_list.append(transforms.ToTensor()(ww_rpn_tmp).to(device))
        else:
            for idx in range(5):
                if idx==1 or idx==2 :
                    rpn_list.append(torch.zeros(w,h).to(device))
                    ww_rpn_list.append(torch.zeros(w,h).to(device))
                else:
                    rpn_list.append(torch.zeros(h,w).to(device))
                    ww_rpn_list.append(torch.zeros(h,w).to(device))
                
        for idx in range(len(image_list)):
            cur = idx*5
            if idx==1 or idx==2 :
                res0 = F.interpolate(res[cur:cur+1], size=(w,h))
                w1 = torch.zeros(w,h).to(device)
                res1 = torch.zeros(w,h).to(device)
            else:
                res0 = F.interpolate(res[cur:cur+1], size=(h,w))
                w1 = torch.zeros(h,w).to(device)
                res1 = torch.zeros(h,w).to(device)
                
            w0 = 2*torch.abs(F.avg_pool2d(res0.sigmoid(), kernel_size=7, stride=1, padding=3)-0.5)
            #w0 = torch.abs(F.avg_pool2d(res0.sigmoid(), kernel_size=7, stride=1, padding=3)-0.7)
            res0 = res0.squeeze()
            w0 = w0.squeeze()
            
            w1[:384,:384] = w1[:384,:384] + ww
            w1[-384:,:384] = w1[-384:,:384] + ww
            w1[:384,-384:] = w1[:384,-384:] + ww
            w1[-384:,-384:] = w1[-384:,-384:] + ww
            
            
            res1[:384,:384] = res1[:384,:384] + res[cur+1,0]
            res1[-384:,:384] = res1[-384:,:384] + res[cur+2,0]
            res1[:384,-384:] = res1[:384,-384:] + res[cur+3,0]
            res1[-384:,-384:] = res1[-384:,-384:] + res[cur+4,0]
            
            
            res_final = (res1+res0*w0+rpn_list[idx]*ww_rpn_list[idx])/(w1+w0+ww_rpn_list[idx])
            #res_final = (res1*(1-w0)+res0*w0)/(w1*(1-w0)+w0)
            res_final = res_final.sigmoid().data.cpu().numpy().squeeze()
            res_final = (res_final>0.5)*1
            
            if idx==1:
                res_final = cv2.rotate(res_final, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif idx==2:
                res_final = cv2.rotate(res_final, cv2.ROTATE_90_CLOCKWISE)
            elif idx==3:
                res_final = cv2.flip(res_final, 0)
            elif idx==4:
                res_final = cv2.flip(res_final, 1)
            final += res_final
        
        final = (final>2)*255
        cv2.imwrite(save_path+'DFUC2022_val_release/'+name, final)
        
        sota = cv2.imread(sota_path+name, cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(ref_path+name, cv2.IMREAD_GRAYSCALE)
        overlap[:,:,0] = sota
        overlap[:,:,1] = final
        axarr[0].imshow(overlap)
        axarr[1].imshow(origin)
        axarr[2].imshow(ref)
        
        plt.show()

# %%
test_path = "../DFUC2022_val_release"
ref_path = "../Val/"
#test_path = "../DFUC2022_train_release/val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
save_path = "/work/u6859530/DFUC/snapshots/crop_with_rpn_all_iou_focal_k7_jitter0.2/"
#save_path = "/work/u6859530/DFUC/snapshots/crop_with_rpn_all_iou_focal_k7/"
sota_path = "/work/u6859530/DFUC/snapshots/TransFuse_B_crop_iou_focal_k7/sota/"
model = TransFuse_B(pretrained=False).to(device)
model = torch.nn.DataParallel(model)
weight = torch.load(save_path+'checkpoint1.pt')
model.load_state_dict(weight['model_state_dict'])

#save_path1 = "/work/u6859530/DFUC/snapshots/TransFuse_B_rpn_k7_jitter0.0/"
#model1 = TransFuse_B(pretrained=False).to(device)
#model1 = torch.nn.DataParallel(model1)
#weight1 = torch.load(save_path1+'best.pt')
#model1.load_state_dict(weight1['model_state_dict']


# %%
print("#"*20, "Start Testing", "#"*20)

#test_(save_path, sota_path, test_path)
test(model, test_path, save_path=save_path, sota_path=sota_path, ref_path=ref_path)
#test_rpn(model, test_path, save_path=save_path, sota_path=sota_path, ref_path=ref_path)


# %%
