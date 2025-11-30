import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# ============================================================
# Label dictionaries (shared everywhere)
# ============================================================

LABELS_DICT = {
    "truck": 0,
    "deer": 1,
    "bird": 2,
    "frog": 3,
    "ship": 4,
    "horse": 5,
    "cat": 6,
    "dog": 7,
    "automobile": 8,
    "airplane": 9,
}

INV_LABELS_DICT = {v: k for k, v in LABELS_DICT.items()}


# ============================================================
# Task 2 training dataset for EfficientNet-B0 (224x224)
# Uses 10 000 labeled images in: ./data/kaggle_challenge/train/
# and ./data/kaggle_challenge/train_labels.csv (id, label)
# ============================================================

class Task2TrainingDataset224(Dataset):
    """
    10k labeled training images (224x224, with light augmentation).
    """

    def __init__(self):
        self.df = pd.read_csv("./data/kaggle_challenge/train_labels.csv")
        self.trainpath = "./data/kaggle_challenge/train/"

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.at[idx, "id"]
        label_str = self.df.at[idx, "label"]

        img_path = os.path.join(self.trainpath, img_id)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = LABELS_DICT[label_str]
        return img, label


# ============================================================
# Task 2 "test-style" dataset (same 10k images, no augmentation)
# Used for validation / CV splits, not for Kaggle test.
# ============================================================

class Task2TestDataset224(Dataset):
    """
    Same 10k train images, but without augmentation.
    Used for validation / cross-validation.
    """

    def __init__(self):
        self.df = pd.read_csv("./data/kaggle_challenge/train_labels.csv")
        self.trainpath = "./data/kaggle_challenge/train/"

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.at[idx, "id"]
        label_str = self.df.at[idx, "label"]

        img_path = os.path.join(self.trainpath, img_id)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = LABELS_DICT[label_str]
        return img, label


# ============================================================
# Kaggle test dataset (2 000 unlabeled images)
# ./data/kaggle_challenge/test/
# Returns (image_tensor, filename) for submission generation.
# ============================================================

class KaggleTestDataset224(Dataset):
    """
    2k unlabeled test images for Kaggle submission.
    """

    def __init__(self):
        self.testpath = "./data/kaggle_challenge/test/"
        # all files in test folder
        self.filenames = sorted(os.listdir(self.testpath))

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.testpath, fname)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        # no label available -> return filename
        return img, fname
