import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from .color_map import COLOR_MAP

def label_to_map(label):
    h,w,_ = label.shape
    label_map = label.reshape(-1,3)
    color_idx = {k:v for k,v in COLOR_MAP.items()}
    mapped = np.array([color_idx[tuple(c)] for c in label_map])
    mapped = mapped.reshape((h,w))
    return torch.from_numpy(mapped).long()

class CamVidDataset(torch.utils.data.Dataset):
    def __init__(self,image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.label_list = sorted(os.listdir(label_dir))
        self.img_transform = transforms.Compose([
            transforms.Resize((360,480)),
            transforms.ToTensor()
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize((360,480), interpolation=Image.NEAREST)
        ])
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        label_name = self.label_list[idx]
        image = Image.open(f'{self.image_dir}/{image_name}').convert("RGB")
        image = self.img_transform(image)        
        label = np.array(self.label_transform(Image.open(f'{self.label_dir}/{label_name}')))
        label_map = label_to_map(label)
        
        return image,label_map