import os, glob

from sklearn.model_selection import train_test_split

import torch

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from PIL import Image

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label


def create_dataLoader(train_dir, test_dir, image_size, batch_size, seed = 42):
    # Load data
    train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
    labels = [path.split('/')[-1].split('.')[0] for path in train_list]

    train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, random_state=seed)
    print(f"Train Data: {len(train_list)}")
    print(f"Validation Data: {len(valid_list)}")
    print(f"Test Data: {len(test_list)}")
    # Set dataset
    train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    val_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    test_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    train_data = CatsDogsDataset(train_list, transform=train_transforms)
    valid_data = CatsDogsDataset(valid_list, transform=val_transforms)
    test_data = CatsDogsDataset(test_list, transform=test_transforms)
    
    # Set datasetLoader
    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader