import os
import torch
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset

default_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale(num_output_channels = 1),
    torchvision.transforms.Normalize((0.5), (0.5)),
    ]
)

class ScaleStuff(object):
    def __init__(self, max = 1400, min = 20):
        self.max = 1400
        self.min = 20
    def __call__(self, item):
        return float(item - self.min)/(self.max - self.min)

default_target_transform = ScaleStuff()

class DropletDataset(Dataset):
    def __init__(self, root_dir, annotations_path, transform = default_transforms, target_transform = default_target_transform):
        self.root_dir = root_dir
        self.annotations_path = annotations_path
        self.csv = pd.read_csv(annotations_path)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.csv.iloc[idx, 0])
        image = Image.open(img_path)
        label = float(self.csv.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

