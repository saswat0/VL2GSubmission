import os
import csv

from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset

class SymbolDataset(Dataset):
    
    def __init__(self, base_path = 'charts', transforms = None):
        
        super().__init__()
        self.base_path = base_path
        self.transforms = transforms
        self.labels = {'vbar_categorical': 0, 'hbar_categorical': 1, 'line': 2, 'pie': 3, 'dot_line': 4}

        self.X = []
        self.y = []

        folder_name = 'train_val'
        csv_path = os.path.join(base_path, "train_val.csv")

        with open(csv_path) as f:
            reader = csv.reader(f)
            next(f)
            for row in reader:
                f_name = os.path.join(base_path, folder_name, row[0]+'.png')
                img = ImageOps.grayscale(Image.open(f_name))

                self.X.append(img)
                self.y.append(row[1])
        
    def __getitem__(self, index):

        image = self.X[index]
        label = self.y[index]

        if self.transforms:
            image = self.transforms(image)
        
        return image, torch.tensor(self.labels[label])
        
    def __len__(self):
        return len(self.X)