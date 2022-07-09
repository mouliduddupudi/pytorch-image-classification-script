import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MyDataSet(Dataset):

    def __init__(self, dataframe, image_dir, transform = None):

        super().__init__()
        
        self.dataframe = dataframe

        self.image_dir = image_dir

        self.transform = transform
        

    def __len__(self):

        return len(self.image_dir)

    
    def __getitem__(self, index):
        
        image_path = self.image_dir[index]

        label_name = image_path.split('/')[-1]

        image = Image.open(image_path)

        label = (self.dataframe[label_name])

        if self.transform is not None:

            image = self.transform(image)

        return (image, label)

