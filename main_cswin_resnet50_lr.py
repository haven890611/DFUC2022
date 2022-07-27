#!/usr/bin/env python
# coding: utf-8
# %%
import torch
from torch.autograd import Variable
from datetime import datetime
from lib.TransFuse_resnet50_CSwin import TransFuse_B
from utils.dataloader import get_loader, test_dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import shutil


# %%
def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


# %%
def test(model, path):
    model.eval()
    
    image_root = '{}/images/'.format(path)
    gt_root = '{}/masks/'.format(path)
    test_loader = test_dataset(image_root, gt_root, testsize=384, filt=True)

    dice_bank = []
    iou_bank = []
    loss_bank = []
    acc_bank = []
    time_bank = []
    """
    for i in range(test_loader.size):
        image, gt= test_loader.load_data()
        image = image.cuda()

        with torch.no_grad():
            start_time = time.perf_counter()
            _, _, res = model(image)
            elapsed_time = time.perf_counter() - start_time
            
        loss = structure_loss(res, gt.cuda())

        res = res.sigmoid().data.cpu().numpy().squeeze()
        gt = 1*(gt>0.5) 
        gt = gt.numpy().squeeze()
        res = 1*(res > 0.5)
        
        smooth = 0.001
        input_flat = np.reshape(res,(-1))
        target_flat = np.reshape(gt,(-1))
        intersection = (input_flat*target_flat)
    
        iou =  ( intersection.sum() + smooth) / (res.sum() + gt.sum() - intersection.sum() + smooth)
        dice =  ( 2*intersection.sum() + smooth) / (res.sum() + gt.sum() + smooth)
        acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

        loss_bank.append(loss.item())
        dice_bank.append(dice)
        iou_bank.append(iou)
        acc_bank.append(acc)
        time_bank.append(elapsed_time)
    """
    for i in range(test_loader.size):
        image, gt, h, w, name= test_loader.load_data()
        image = image.cuda()
        with torch.no_grad():
            start_time = time.perf_counter()
            _, _, res = model(image)
            elapsed_time = time.perf_counter() - start_time
        
        res = F.interpolate(res, size=(h,w))
        loss = structure_loss(res, gt.cuda())
        
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
        
        loss_bank.append(loss.item())
        dice_bank.append(dice)
        iou_bank.append(iou)
        acc_bank.append(acc)
        time_bank.append(elapsed_time)
        
    print('Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, fps: {:.4f}'.
        format(np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank), 1./np.mean(time_bank)))
    
    
    total_vali_loss_history.append(np.mean(loss_bank))
    vali_accuracy_history.append(np.mean(acc_bank))
    dice_history.append(np.mean(dice_bank))
    iou_history.append(np.mean(iou_bank))
    
    return np.mean(loss_bank), np.mean(acc_bank), np.mean(dice_bank), np.mean(iou_bank), 1./np.mean(time_bank)


# %%
best = 0.0
def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter()
    total_loss = 0.
    
    for i, pack in tqdm(enumerate(train_loader, start=1), desc=str(epoch)):
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # ---- forward ----
        lateral_map_4, lateral_map_3, lateral_map_2 = model(images)  

        # ---- loss function ----
        loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)
        loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4  

        total_loss += loss2
        # ---- backward ----
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)  
        optimizer.step()
        optimizer.zero_grad()

        # ---- recording loss ----
        loss_record2.update(loss2.data, batch_size)
        loss_record3.update(loss3.data, batch_size)
        loss_record4.update(loss4.data, batch_size)


    print("Epoch" + str(epoch))
    save_path = '/work/u6859530/DFUC/snapshots/{}/'.format(train_save)
    os.makedirs(save_path, exist_ok=True)
    print('Saving checkpoint...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch':epoch,
        }, save_path + 'checkpoint.pt')
    
    meanloss, mean_acc, mean_dice, mean_iou, mean_fps = test(model, vali_path)
    global best
    if mean_dice > best:
        print('[Saving Snapshot:]', save_path + 'best.pt   Dice: {:0.4f}'.format(mean_dice))
        best = mean_dice
        shutil.copyfile(save_path + 'checkpoint.pt', save_path + 'best.pt')
        
    if epoch == 1 :
        fp = open(save_path + 'record.txt','w')
    else:
        fp = open(save_path + 'record.txt','a')
    fp.write('{} {:0.4f} {:0.4f} {:0.4f}\n'.format(epoch, mean_iou, mean_dice, total_loss/len(train_loader)))


# %%
# hyperparameters
epochs = 100
lr = 3e-5
batch_size = 8
grad_norm =2.0
beta1 = 0.5
beta2 = 0.99

train_path = "/home/u6859530/DFUC/DFUC2022_train_release/train"
vali_path = "/home/u6859530/DFUC/DFUC2022_train_release/val"

train_image_root = '{}/images/'.format(train_path)
train_gt_root = '{}/masks/'.format(train_path)

train_save = 'TransFuse_B_lr_filter'


# %%
total_vali_loss_history = []
vali_accuracy_history = []
dice_history = []
iou_history = []


# %%
# Create device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransFuse_B(pretrained=True)
model = torch.nn.DataParallel(model)
#weight = torch.load('/home/u6859530/DFUC/Transfuse_v2/snapshots/TransFuse_B_polyp_pretrain/checkpoint.pt')
#model.load_state_dict(weight['model_state_dict'], strict=True)


# %%
my_list = model.named_parameters()
params = []
resnet_params = []
base_params = []
for n, p in my_list:
    #if n.startswith('module.transformer.'):
    #    params.append(p)
    if n.startswith('module.resnet.'):
        resnet_params.append(p)
    else:
        base_params.append(p)

optimizer = torch.optim.AdamW([
                {'params': resnet_params, 'lr': 1e-4},
                {'params': base_params}
            ], 3e-5, betas=(beta1, beta2))

#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-5,last_epoch=-1)

# %%
train_loader = get_loader(train_image_root, train_gt_root, batchsize=batch_size, img_size=384, filt=True)
total_step = len(train_loader)

print("#"*20, "Start Training", "#"*20)

for epoch in range(1, epochs + 1):
    train(train_loader, model, optimizer, epoch)
    #scheduler.step()


# %%
