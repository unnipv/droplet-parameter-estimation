import os
import torch
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#Change values according to dataset
MAX_DIA = 400
MIN_DIA = 50

default_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale(num_output_channels = 1),
    torchvision.transforms.Normalize((0.5), (0.5)),
    ]
)

class ScaleStuff(object):
    def __init__(self, max_dia = MAX_DIA, min_dia = MIN_DIA):
        self.max_dia = MAX_DIA
        self.min_dia = MIN_DIA
    def __call__(self, item):
        dia_normalized = float(item - self.min_dia)/(self.max_dia - self.min_dia)
        return torch.Tensor([dia_normalized]).squeeze()

default_target_transform = ScaleStuff()

class DropletDataset(Dataset):
    def __init__(self, root_dir, annotations_path, transform = default_transforms, target_transform = default_target_transform):
        self.root_dir = root_dir
        self.annotations_path = annotations_path
        self.csv = pd.read_csv(annotations_path)
#Uncomment this if resampling is required in case of very large dataset
        # self.csv = self.csv.sample(frac = 0.6, random_state = 42)
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

#Debug lines
if __name__ == "__main__":
    ROOT_DIR = r"data"
    TEST_CSV = r"data\test.csv"
    test_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = TEST_CSV)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    for data, target in test_dataloader:
        print(target)
        print("Diameters:")
        print(str(50 + target * 350))

