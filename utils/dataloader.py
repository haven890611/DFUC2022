import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
from timm.data.auto_augment import rand_augment_transform


MEAN = [0.5572, 0.3216, 0.2357] 
STD = [0.3060, 0.2145, 0.1773]


class KvasirDataset(data.Dataset):
    """
    dataloader for skin lesion segmentation tasks
    """
    def __init__(self, image_root, gt_root, img_size, jitter=0.5, filt=False):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        # self.images = np.load(image_root, allow_pickle=True)
        # self.gts = np.load(gt_root, allow_pickle=True)
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images.sort()
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts.sort(key=lambda x: x.replace("_mask", ""))
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize(MEAN, STD)
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize(img_size),
        ])
        self.ff_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
        ])

        self.transform = A.Compose([
            #rand_aug,
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
            A.ColorJitter(jitter, jitter, jitter),
            A.HorizontalFlip(),
            A.VerticalFlip()
        ])

        self.filt = filt
    def __getitem__(self, index):       
        image_path = self.images[index]
        gt_path = self.gts[index]
        image = cv2.imread(image_path)
        gt = cv2.imread(gt_path)
        gt = gt/255.0
        h, w, _ = image.shape
        
        if self.filt:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = np.ones((h,w))*128
            im2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            ff = 1.*((im2[:,:,0:1]*0.5+im2[:,:,1:2]*0.5)<180)

            transformed = self.transform(image=image, mask=gt, filt=ff)
            image = self.img_transform(transformed['image'])
            filter = self.ff_transform(transformed['filt']).float()
            gt = self.gt_transform(transformed['mask'])
            image = image*filter
            
        else:
            transformed = self.transform(image=image, mask=gt)
            image = self.img_transform(transformed['image'])
            gt = self.gt_transform(transformed['mask'])
            
        return image, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, img_size=384, shuffle=True, num_workers=4, pin_memory=True, jitter=0.5, filt=False):

    dataset = KvasirDataset(image_root, gt_root, img_size=img_size, jitter=jitter, filt=filt)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root=None, testsize=384, hr=False, origin=False, filt=False):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]    
        self.images = sorted(self.images)
        if hr:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((testsize, testsize)),
                transforms.Normalize(MEAN, STD)
            ])
        self.gts = None
        if gt_root is not None:
            self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gts = sorted(self.gts)
            self.gt_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale()
            ])
            
        self.size = len(self.images)
        self.index = 0
        self.origin = origin
        self.filt = filt
        self.ff_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((testsize, testsize))
        ])
            
        
    def __len__(self):
        return self.size
    
    def load_data(self):
        image_path = self.images[self.index]
        origin_img = cv2.imread(image_path)
        h, w, _ = origin_img.shape 
        if self.filt:
            hsv = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = np.ones((h,w))*128
            im2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            ff = 1.*((im2[:,:,0:1]*0.5+im2[:,:,1:2]*0.5)<180)
            image = (self.transform(origin_img)*self.ff_transform(ff).float()).unsqueeze(0)
        else:
            image = self.transform(origin_img).unsqueeze(0)
        
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        
        if self.gts:
            gt_path = self.gts[self.index-1]
            gt = cv2.imread(gt_path)
            gt = self.gt_transform(gt).unsqueeze(0)
            return image, gt, h, w, name
        else:
            if self.origin:
                return image, h, w, name, cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
            else:
                return image, h, w, name

if __name__ == '__main__':
    path = 'data/'
    tt = KvasirDataset(path+'images.npy', path+'masks.npy')

    for i in range(50):
        img, gt = tt.__getitem__(i)

        img = torch.transpose(img, 0, 1)
        img = torch.transpose(img, 1, 2)
        img = img.numpy()
        gt = gt.numpy()

        plt.imshow(img)
        plt.savefig('vis/'+str(i)+".jpg")
 
        plt.imshow(gt[0])
        plt.savefig('vis/'+str(i)+'_gt.jpg')
