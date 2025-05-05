import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ICLEVRDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None, mode='train'):
        """
        Args:
            json_path (string): Path to the json file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train' or 'test' mode
        """
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        
        # Load object mapping
        with open(os.path.join(os.path.dirname(json_path), 'objects.json'), 'r') as f:
            self.object_mapping = json.load(f)
        
        # Load data
        with open(json_path, 'r') as f:
            if mode == 'train':
                self.data = json.load(f)
                self.filenames = list(self.data.keys())
            else:  # test mode
                self.data = json.load(f)
                self.filenames = [f"test_{i}" for i in range(len(self.data))]
        
        self.num_classes = len(self.object_mapping)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            img_name = self.filenames[idx]
            img_path = os.path.join(self.image_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            
            # Create one-hot label
            label = torch.zeros(self.num_classes)
            for obj in self.data[img_name]:
                label[self.object_mapping[obj]] = 1
        else:  # test mode
            # For test mode, we only return the label
            label = torch.zeros(self.num_classes)
            for obj in self.data[idx]:
                label[self.object_mapping[obj]] = 1
            
            # Create a dummy image (will not be used)
            image = Image.new('RGB', (64, 64), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloader(json_path, image_dir, batch_size, mode='train', shuffle=True):
    """
    Create a dataloader for the ICLEVR dataset
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = ICLEVRDataset(json_path, image_dir, transform, mode)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=16,
        pin_memory=True
    )
    
    return dataloader
