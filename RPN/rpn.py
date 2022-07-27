import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.io import read_image
#from torchvision.ops import masks_to_boxes, box_convert
from torchvision.utils import draw_bounding_boxes, save_image
from torchvision.transforms.functional import crop
from torch.utils.data import DataLoader

# +
train_mask_dir = "../DFUC2022_train_release/train/masks/"
train_img_dir = "../DFUC2022_train_release/train/images/"
val_mask_dir = "../DFUC2022_train_release/val/masks/"
val_img_dir = "../DFUC2022_train_release/val/images/"

test_img_dir = '../DFUC2022_val_release/'
image_size = (480, 640)


# -

def listdir(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


# +
plt.rcParams["savefig.bbox"] = "tight"

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


# +
#class DetectionDataset(torch.utils.data.Dataset):
#    def __init__(self, root, transforms):
#        self.root = root # ../DFUC2022_train_release/train/
#        self.transforms = transforms
#        # load all image files, sorting them to
#        # ensure that they are aligned
#        self.imgs = list(sorted(listdir(root + 'images')))
#        self.masks = list(sorted(listdir(root + 'masks')))
#    
#    def __len__(self):
#        return len(self.imgs)
#    
#    def __getitem__(self, idx):
#        # load images and masks
#        img_path = os.path.join(self.root + 'images', self.imgs[idx])
#        mask_path = os.path.join(self.root + 'masks', self.masks[idx])
#
#        img = read_image(img_path)
#        mask = read_image(mask_path)
#
#        img = F.convert_image_dtype(img, dtype=torch.float)
#        mask = F.convert_image_dtype(mask, dtype=torch.float)
#
#        # We get the unique colors, as these would be the object ids.
#        obj_ids = torch.unique(mask)
#
#        # first id is the background, so remove it.
#        obj_ids = obj_ids[1:]
#
#        # split the color-encoded mask into a set of boolean masks.
#        masks = mask == obj_ids[:, None, None]
#
#        boxes = masks_to_boxes(masks)
#        boxes = boxes[(boxes[:,0] < boxes[:,2]) & (boxes[:,1] < boxes[:,3])]
#        # ground truth bouding box sizes:
#        # print(boxes[:, 2:4] - boxes[:, 0:2])
#        
#        # there is only one class
#        labels = torch.ones((masks.shape[0],), dtype=torch.int64)
#
#        target = {}
#        target["boxes"] = boxes
#        target["labels"] = labels
#
#        #if self.transforms is not None:
#            #img, target = self.transforms(img, target)
#
#        return img, target

# +
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# -

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# +
from torchvision.transforms.functional import convert_image_dtype

score_threshold = .9

model.load_state_dict(torch.load('models/10000.torch'))

model.eval()

# +
# super bounding box: [min_x, min_y, max_x, max_y]
# (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
# x1: left, x2: right, y1: lower, y2: upper

val_top_lefts = []

for img_name, mask_name in zip(list(sorted(listdir(val_img_dir))), list(sorted(listdir(val_mask_dir)))):
    img = read_image(val_img_dir + img_name)
    mask = read_image(val_mask_dir + mask_name)
    batch = convert_image_dtype(img, dtype=torch.float)
    batch = batch.to(device)

    outputs = model(batch.unsqueeze(0))
    for output in outputs:
        boxes = output['boxes'][output['scores'] > score_threshold]
        box = []
        
        if len(boxes):
            xmin = torch.min(boxes[:,0]).item()
            ymin = torch.min(boxes[:,1]).item()
            xmax = torch.max(boxes[:,2]).item()
            ymax = torch.max(boxes[:,3]).item()
            
            if xmax - xmin < 384:
                xmid = (xmin + xmax) / 2
                xmin = xmid - 192
                xmax = xmid + 192
                if xmin < 0:
                    xmin = 0
                    xmax = 384
                if xmax > 640:
                    xmax = 640
                    xmin = 640 - 384
            #else:
                #print(img_name, xmax - xmin, ymax - ymin)
            
            if ymax - ymin < 384:
                ymid = (ymin + ymax) / 2
                ymin = ymid - 192
                ymax = ymid + 192
                if ymin < 0:
                    ymin = 0
                    ymax = 384
                if ymax > 480:
                    ymax = 480
                    ymin = 480 - 384
            #else:
                #print(img_name, xmax - xmin, ymax - ymin)
                
            box = [xmin, ymin, xmax, ymax]
        
        box = torch.as_tensor(box)

    for output in outputs:
        if len(box):
            #drawn_box = draw_bounding_boxes(img, boxes=box.unsqueeze(0),
            #                                width=4,
            #                                colors='green')
            val_top_lefts.append([img_name, int(box[1]), int(box[0])])
            cropped_img = crop(img, int(box[1]), int(box[0]), 384, 384)
            cropped_mask = crop(mask, int(box[1]), int(box[0]), 384, 384)
            save_image(cropped_img.to(torch.float)/255, 'RP/val/images/' + img_name)
            save_image(cropped_mask.to(torch.float)/255, 'RP/val/masks/' + mask_name)

# +
# testing image cropping

test_top_lefts = []

for img_name in list(sorted(listdir(test_img_dir))):
    print(img_name)
    img = read_image(test_img_dir + img_name)
    batch = convert_image_dtype(img, dtype=torch.float)
    batch = batch.to(device)

    outputs = model(batch.unsqueeze(0))
    for output in outputs:
        boxes = output['boxes'][output['scores'] > score_threshold]
        box = []
        
        if len(boxes):
            xmin = torch.min(boxes[:,0]).item()
            ymin = torch.min(boxes[:,1]).item()
            xmax = torch.max(boxes[:,2]).item()
            ymax = torch.max(boxes[:,3]).item()
            
            if xmax - xmin < 384:
                xmid = (xmin + xmax) / 2
                xmin = xmid - 192
                xmax = xmid + 192
                if xmin < 0:
                    xmin = 0
                    xmax = 384
                if xmax > 640:
                    xmax = 640
                    xmin = 640 - 384
            #else:
                #print(img_name, xmax - xmin, ymax - ymin)
            
            if ymax - ymin < 384:
                ymid = (ymin + ymax) / 2
                ymin = ymid - 192
                ymax = ymid + 192
                if ymin < 0:
                    ymin = 0
                    ymax = 384
                if ymax > 480:
                    ymax = 480
                    ymin = 480 - 384
            #else:
                #print(img_name, xmax - xmin, ymax - ymin)
                
            box = [xmin, ymin, xmax, ymax]
        
        box = torch.as_tensor(box)

    for output in outputs:
        if len(box):
            #drawn_box = draw_bounding_boxes(img, boxes=box.unsqueeze(0),
            #                                width=4,
            #                                colors='green')
            #show(drawn_box)
            test_top_lefts.append([img_name, int(box[1]), int(box[0])])
            cropped_img = crop(img, int(box[1]), int(box[0]), 384, 384)
            save_image(cropped_img.to(torch.float)/255, 'RP/test/' + img_name)

# +
import csv

with open('test_top_lefts.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(test_top_lefts)
# -


