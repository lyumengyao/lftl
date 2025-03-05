import argparse
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
import cv2
import torchvision
from torch.utils.data import DataLoader

def make_dataset(image_list, dset_root):
    assert isinstance(image_list, list) and len(image_list) != 0
    images = []
    for line in image_list:
        item = line.strip().split()
        img_path = os.path.join(dset_root, item[0])
        img_label = int(item[1])
        images.append([img_path, img_label])
        assert os.path.exists(img_path), "[WRONG] %s does not exist" % img_path
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, imgs, transform=None,  mode='RGB'):
        
        self.imgs = imgs
        self.transform = transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index

    def __len__(self):
        return len(self.imgs)

def get_dataset(dset_file, dset_root): 
    ## prepare data
    txt_dset = open(dset_file).readlines()
    return make_dataset(txt_dset, dset_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data set')
    
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    args = parser.parse_args()

    dset_file_path = "./data/train_list.txt"
    mode="train" #validation
    args.dset_root = os.path.join("/home/chenhui/Code/dataset/VISDA-C", mode)
    dataset = get_dataset(dset_file_path, args.dset_root)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=False)

    print("load %d images from %s"%(len(dataset), args.dset_root))

    dset_file_path = "./data/validation_list.txt"
    mode="validation"
    args.dset_root = os.path.join("/home/chenhui/Code/dataset/VISDA-C", mode)
    dataset = get_dataset(dset_file_path, args.dset_root)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=False)

    print("load %d images from %s"%(len(dataset), args.dset_root))
    