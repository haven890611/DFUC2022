import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes, box_convert
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader

train_mask_dir = "../DFUC2022_train_release/train/masks/"
train_img_dir = "../DFUC2022_train_release/train/images/"
val_mask_dir = "../DFUC2022_train_release/val/masks/"
val_img_dir = "../DFUC2022_train_release/val/images/"
image_size = (480, 640)


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


# -

class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root # ../DFUC2022_train_release/train/
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(listdir(root + 'images')))
        self.masks = list(sorted(listdir(root + 'masks')))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root + 'images', self.imgs[idx])
        mask_path = os.path.join(self.root + 'masks', self.masks[idx])

        img = read_image(img_path)
        mask = read_image(mask_path)

        img = F.convert_image_dtype(img, dtype=torch.float)
        mask = F.convert_image_dtype(mask, dtype=torch.float)

        # We get the unique colors, as these would be the object ids.
        obj_ids = torch.unique(mask)

        # first id is the background, so remove it.
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of boolean masks.
        masks = mask == obj_ids[:, None, None]

        boxes = masks_to_boxes(masks)
        boxes = boxes[(boxes[:,0] < boxes[:,2]) & (boxes[:,1] < boxes[:,3])]
        # ground truth bouding box sizes:
        # print(boxes[:, 2:4] - boxes[:, 0:2])
        
        # there is only one class
        labels = torch.ones((masks.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        #if self.transforms is not None:
            #img, target = self.transforms(img, target)

        return img, target


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
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
batch_size = 32

model.train()
for i in range(10001):
    images = []
    targets = []
    dataset = DetectionDataset('../DFUC2022_train_release/train/', None)
    
    for b in range(batch_size):
        rand = random.randint(0,1799)
        image, target = dataset[rand]
        images.append(image)
        targets.append(target)
    
    images = list(image.to(device) for image in images)
    targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
   
    optimizer.zero_grad()
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
   
    losses.backward()
    optimizer.step()
   
    print(i,'loss:', losses.item())
    if i%200==0:
        torch.save(model.state_dict(), str(i)+".torch")
        print("Save model to:",str(i)+".torch")

# +
from torchvision.transforms.functional import convert_image_dtype

score_threshold = .9

model.load_state_dict(torch.load('10000.torch'))

model.eval()
# -

# super bounding box: [min_x, min_y, max_x, max_y]
# (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
# x1: left, x2: right, y1: lower, y2: upper
for img_name in list(sorted(listdir(val_img_dir))):
    img = read_image(val_img_dir + img_name)
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
            box = [xmin, ymin, xmax, ymax]
        box = torch.as_tensor(box)

    for output in outputs:
        if len(box):
            drawn_box = draw_bounding_boxes(img, boxes=box.unsqueeze(0),
                                            width=4,
                                            colors='green')
            show(drawn_box)

# +
# c: [cx, cy, w, h]
box_dict = []
for center in centers:
    boxes = []
    for c in center['boxes']:
        box = torch.stack((c[0] - 192, c[1] + 192))
        if torch.all(box >= 0):
            print(box)
            boxes.append(box)
    
    box = {}
    box['filename'] = center['filename']
    box['boxes'] = boxes
    box_dict.append(box)

print(box_dict)
# -

from torchvision.transforms.functional import crop
for boxes in box_dict:
    for box in boxes['boxes']:
        # print(int(box[0].item()),int(box[1].item()))
        cropped = crop(read_image(val_img_dir + boxes['filename']),
             int(box[0].item()),
             int(box[1].item()),
             384,
             384)
        show(cropped)

# +
import csv

with open('centers.csv', 'w') as f:
    dict_writer = csv.DictWriter(f, centers[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(centers)

# +
# loading and repurposing dataset
from utils.dataloader import get_loader, test_dataset

train_loader = get_loader(train_img_dir, train_mask_dir, batchsize=16, img_size=image_size)
dataset = DetectionDataset('../DFUC2022_train_release/train/', None)
imgs = []
targets = []
for img, target in dataset:
    imgs.append(img)
    targets.append(target)
print(len(targets))

# +
# drawing ground truth bounding boxes

drawn_boxes_list = []
for img_name in listdir(mask_dir):
    img = read_image(mask_dir + img_name)
    print(img.type())
    boxes = torchvision.ops.masks_to_boxes(img)
    drawn_boxes = draw_bounding_boxes(img, boxes, colors='red')
    drawn_boxes_list.append(drawn_boxes)
    show(drawn_boxes)

# +
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from utils.dataloader import get_loader, test_dataset

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights, progress=False)

img = read_image(train_img_dir + '100001.jpg')
mask = read_image(train_mask_dir + '100001.png')

img = F.convert_image_dtype(img, dtype=torch.float)
mask = F.convert_image_dtype(mask, dtype=torch.float)

obj_ids = torch.unique(mask)
obj_ids = obj_ids[1:]
masks = mask == obj_ids[:, None, None]
boxes = masks_to_boxes(masks)

# training
target = {}
target['boxes'] = boxes
target['labels'] = torch.ones((masks.shape[0],), dtype=torch.int64)
print(target['boxes'])
output = model(img.unsqueeze(0), [target])

model.eval()
img = read_image(train_img_dir + '100003.jpg')
batch = convert_image_dtype(img, dtype=torch.float)

outputs = model(batch.unsqueeze(0))
print(outputs)
