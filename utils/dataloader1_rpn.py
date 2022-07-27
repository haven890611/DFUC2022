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
    def __init__(self, image_root, gt_root, img_size):
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

        self.transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
            A.ColorJitter(0.5, 0.5, 0.5),
            A.HorizontalFlip(),
            A.VerticalFlip()
        ])


    def __getitem__(self, index):
        print(index)
        image_path = self.images[index]
        gt_path = self.gts[index]
        image = cv2.imread(image_path)
        gt = cv2.imread(gt_path)
        gt = gt/255.0
        
        index = random.randint(1,4)
        if index == 4:
            image1 = image[:384,:384,:]
            gt1 = gt[:384,:384,:]
        elif index == 3:
            image1 = image[-384:,:384,:]
            gt1 = gt[-384:,:384,:]
        elif index == 2:
            image1 = image[:384,-384:,:]
            gt1 = gt[:384,-384:,:]
        else:
            image1 = image[-384:,-384:,:]
            gt1 = gt[-384:,-384:,:]
        
        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = self.gt_transform(transformed['mask'])
        
        transformed1 = self.transform(image=image1, mask=gt1)
        image1 = self.img_transform(transformed1['image'])
        gt1 = self.gt_transform(transformed1['mask'])
        
        return image, gt, image1, gt1

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, img_size=384, shuffle=True, num_workers=4, pin_memory=True):

    dataset = KvasirDataset(image_root, gt_root, img_size=img_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root=None, testsize=384, hr=False, origin=False):
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
        self.index += 1
        
        if self.gts:
            gt_list = []
            gt_path = self.gts[self.index-1]
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
            return image_list, h, w, name


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
