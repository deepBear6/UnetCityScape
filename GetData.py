import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import os
import random
import glob
import numpy as np

class SegmentationDataset(Dataset):

    def __init__(self, image_dir, label_dir, transform=None):
        self.image_names = []
        self.label_names = []
        for folder in os.listdir(image_dir):
            for image in os.listdir(os.path.join(image_dir, folder)):
                self.image_names.append(os.path.join(image_dir, folder, image))
        for folder in os.listdir(label_dir):
            for image in glob.glob(os.path.join(label_dir, folder,"*labelIds.png")):
                self.label_names.append(os.path.join(image))

        self.image_names.sort()
        self.label_names.sort()
        self.transform = transform
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = Image.open(self.image_names[idx])
        label = Image.open(self.label_names[idx])
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        image_tensor = transforms.ToTensor()(image)
        image_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image_tensor)
        #label = transforms.ToTensor()(label)
        #height, width, channels = np.array(image).shape
        #image_tensor = torch.from_numpy(np.array(image)).view(channels, height, width).float()
        label = torch.from_numpy(np.array(label)).long()
        image = torch.from_numpy(np.array(image))
        return image, image_tensor, label
