import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.to_tensor = transforms.ToTensor()
        self.grayscale = transforms.Grayscale(num_output_channels=1)
        self.filenames = os.listdir(data_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.data_dir, filename)
        img = Image.open(img_path)
        gray_img = self.grayscale(img)
        img_tensor = self.to_tensor(img)
        gray_tensor = self.to_tensor(gray_img)

        return gray_tensor.float(), img_tensor.float()
