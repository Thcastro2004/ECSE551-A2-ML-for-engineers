import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class TrainingDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv('./data/kaggle_challenge/train_labels.csv')
        self.trainpath = './data/kaggle_challenge/train/'
        self.transform = transforms.Compose([
            transforms.RandomApply(nn.ModuleList([transforms.RandomResizedCrop(size=(32,32), scale=(0.8,1))]),p=0.1),
            transforms.RandomApply(nn.ModuleList([transforms.RandomRotation((1,5))]),p=0.1),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomApply(nn.ModuleList([transforms.ColorJitter((0.7,1),(0.7,1),(0.7,1),(-0.1,0.1))]),p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        self.labels_dict = {
            "truck": 0,
            "deer": 1,
            "bird": 2,
            "frog": 3,
            "ship": 4,
            "horse": 5,
            "cat": 6,
            "dog": 7,
            "automobile": 8,
            "airplane": 9
        }
        return

    def __getitem__(self, idx):
        img = Image.open(self.trainpath+self.df.at[idx, "id"])
        img = img.convert("RGB")
        img = self.transform(img)
        label = self.labels_dict.get(self.df.at[idx, "label"])
        return img, label

    def __len__(self):
        return len(self.df)


class TestingDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv('./data/kaggle_challenge/train_labels.csv')
        self.trainpath = './data/kaggle_challenge/train/'
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        self.labels_dict = {
            "truck": 0,
            "deer": 1,
            "bird": 2,
            "frog": 3,
            "ship": 4,
            "horse": 5,
            "cat": 6,
            "dog": 7,
            "automobile": 8,
            "airplane": 9
        }
        return

    def __getitem__(self, idx):
        img = Image.open(self.trainpath+self.df.at[idx, "id"])
        img = img.convert("RGB")
        img = self.transform(img)
        label = self.labels_dict.get(self.df.at[idx, "label"])
        return img, label

    def __len__(self):
        return len(self.df)


class Test551(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv('./data/kaggle_challenge/sample_submission.csv')
        self.trainpath = './data/kaggle_challenge/test/'
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        self.labels_dict = {
            "truck": 0,
            "deer": 1,
            "bird": 2,
            "frog": 3,
            "ship": 4,
            "horse": 5,
            "cat": 6,
            "dog": 7,
            "automobile": 8,
            "airplane": 9
        }
        return

    def __getitem__(self, idx):
        img = Image.open(self.trainpath+self.df.at[idx, "id"])
        img = img.convert("RGB")
        img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.df)


class Dataset551(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv('./data/kaggle_challenge/sample_submission.csv')
        self.trainpath = './data/kaggle_challenge/test/'
        self.transform = transforms.ToTensor()
        self.labels_dict = {
            "truck": 0,
            "deer": 1,
            "bird": 2,
            "frog": 3,
            "ship": 4,
            "horse": 5,
            "cat": 6,
            "dog": 7,
            "automobile": 8,
            "airplane": 9
        }
        return

    def __getitem__(self, idx):
        img = Image.open(self.trainpath+self.df.at[idx, "id"])
        img = img.convert("RGB")
        img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.df)


class Task2TrainingDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv('./data/kaggle_challenge/train_labels.csv')
        self.trainpath = './data/kaggle_challenge/train/'
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(32, 32), scale=(0.7, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.labels_dict = {
            "truck": 0, "deer": 1, "bird": 2, "frog": 3, "ship": 4,
            "horse": 5, "cat": 6, "dog": 7, "automobile": 8, "airplane": 9
        }
    
    def __getitem__(self, idx):
        img = Image.open(self.trainpath + self.df.at[idx, "id"])
        img = img.convert("RGB")
        img = self.transform(img)
        label = self.labels_dict.get(self.df.at[idx, "label"])
        return img, label
    
    def __len__(self):
        return len(self.df)


class Task2TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = pd.read_csv('./data/kaggle_challenge/train_labels.csv')
        self.trainpath = './data/kaggle_challenge/train/'
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.labels_dict = {
            "truck": 0, "deer": 1, "bird": 2, "frog": 3, "ship": 4,
            "horse": 5, "cat": 6, "dog": 7, "automobile": 8, "airplane": 9
        }
    
    def __getitem__(self, idx):
        img = Image.open(self.trainpath + self.df.at[idx, "id"])
        img = img.convert("RGB")
        img = self.transform(img)
        label = self.labels_dict.get(self.df.at[idx, "label"])
        return img, label
    
    def __len__(self):
        return len(self.df)

