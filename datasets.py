import os
import pickle
import numpy as np
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
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.02,
                ),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
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


# ============================================================
# Task 2 training dataset for EfficientNet-B3 (300x300)
# Uses 10 000 labeled images in: ./data/kaggle_challenge/train/
# and ./data/kaggle_challenge/train_labels.csv (id, label)
# ============================================================

class Task2TrainingDataset300(Dataset):
    """
    10k labeled training images (300x300, with light augmentation).
    For EfficientNet-B3.
    """

    def __init__(self):
        self.df = pd.read_csv("./data/kaggle_challenge/train_labels.csv")
        self.trainpath = "./data/kaggle_challenge/train/"

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(300, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.02,
                ),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
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
# Task 2 "test-style" dataset (same 10k images, no augmentation, 300x300)
# Used for validation / CV splits, not for Kaggle test.
# ============================================================

class Task2TestDataset300(Dataset):
    """
    Same 10k train images, but without augmentation (300x300).
    Used for validation / cross-validation.
    For EfficientNet-B3.
    """

    def __init__(self):
        self.df = pd.read_csv("./data/kaggle_challenge/train_labels.csv")
        self.trainpath = "./data/kaggle_challenge/train/"

        self.transform = transforms.Compose(
            [
                transforms.Resize(300),
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
# Kaggle test dataset for EfficientNet-B3 (300x300)
# ./data/kaggle_challenge/test/
# Returns (image_tensor, filename) for submission generation.
# ============================================================

class KaggleTestDataset300(Dataset):
    """
    2k unlabeled test images for Kaggle submission (300x300).
    For EfficientNet-B3.
    """

    def __init__(self):
        self.testpath = "./data/kaggle_challenge/test/"
        # all files in test folder
        self.filenames = sorted(os.listdir(self.testpath))

        self.transform = transforms.Compose(
            [
                transforms.Resize(300),
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


# ============================================================
# Task 2 training dataset (32x32, no resizing)
# Uses 10 000 labeled images in: ./data/kaggle_challenge/train/
# and ./data/kaggle_challenge/train_labels.csv (id, label)
# Images are kept at their original 32x32 size
# ============================================================

class Task2TrainingDataset32(Dataset):
    """
    10k labeled training images (32x32, with light augmentation, no resizing).
    Images are used at their original resolution.
    """

    def __init__(self):
        self.df = pd.read_csv("./data/kaggle_challenge/train_labels.csv")
        self.trainpath = "./data/kaggle_challenge/train/"

        self.transform = transforms.Compose(
            [
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
# Task 2 "test-style" dataset (32x32, no augmentation, no resizing)
# Used for validation / CV splits, not for Kaggle test.
# Images are kept at their original 32x32 size
# ============================================================

class Task2TestDataset32(Dataset):
    """
    Same 10k train images, but without augmentation (32x32, no resizing).
    Used for validation / cross-validation.
    Images are used at their original resolution.
    """

    def __init__(self):
        self.df = pd.read_csv("./data/kaggle_challenge/train_labels.csv")
        self.trainpath = "./data/kaggle_challenge/train/"

        self.transform = transforms.Compose(
            [
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
# CIFAR-10 batch files dataset (300x300)
# Loads data from cifar-10-batches-py/data_batch_* files
# For EfficientNet-B3.
# ============================================================

class CIFAR10BatchDataset300(Dataset):
    """
    Loads CIFAR-10 batch files (data_batch_1 through data_batch_5).
    Images are resized to 300x300 for EfficientNet-B3.
    Uses lazy loading to avoid memory issues.
    """
    
    # CIFAR-10 label order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    # Maps CIFAR-10 label index to our LABELS_DICT index
    CIFAR10_TO_LABELS = {
        0: 9,  # airplane -> airplane
        1: 8,  # automobile -> automobile
        2: 2,  # bird -> bird
        3: 6,  # cat -> cat
        4: 1,  # deer -> deer
        5: 7,  # dog -> dog
        6: 3,  # frog -> frog
        7: 5,  # horse -> horse
        8: 4,  # ship -> ship
        9: 0,  # truck -> truck
    }

    def __init__(self, batch_dir="./cifar-10-batches-py"):
        self.batch_dir = batch_dir
        self.batch_files = []
        self.batch_cache = {}  # Cache loaded batches
        self.batch_offsets = []  # Starting index for each batch
        
        # Find all batch files (CIFAR-10 batches are always 10k samples each)
        current_offset = 0
        for i in range(1, 6):
            batch_file = os.path.join(batch_dir, f"data_batch_{i}")
            if os.path.exists(batch_file):
                self.batch_files.append(batch_file)
                self.batch_offsets.append(current_offset)
                current_offset += 10000  # Each CIFAR-10 batch has 10k samples
        
        if not self.batch_files:
            raise FileNotFoundError(f"No CIFAR-10 batch files found in {batch_dir}")
        
        self.total_samples = len(self.batch_files) * 10000
        
        self.transform = transforms.Compose([
            transforms.Resize(300),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def _load_batch(self, batch_idx):
        """Load a batch file if not already cached."""
        if batch_idx not in self.batch_cache:
            batch_file = self.batch_files[batch_idx]
            with open(batch_file, 'rb') as f:
                batch_dict = pickle.load(f, encoding='bytes')
                if b'data' in batch_dict:
                    batch_data = batch_dict[b'data']
                    batch_labels = batch_dict[b'labels']
                else:
                    batch_data = batch_dict['data']
                    batch_labels = batch_dict['labels']
                
                # Reshape from (N, 3072) to (N, 32, 32, 3)
                batch_data = batch_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                # Convert labels to our label mapping
                batch_labels = [self.CIFAR10_TO_LABELS[label] for label in batch_labels]
                
                self.batch_cache[batch_idx] = (batch_data, batch_labels)
        
        return self.batch_cache[batch_idx]
    
    def _get_batch_and_idx(self, global_idx):
        """Convert global index to batch index and local index."""
        for batch_idx, offset in enumerate(self.batch_offsets):
            if global_idx < offset + 10000:
                local_idx = global_idx - offset
                return batch_idx, local_idx
        raise IndexError(f"Index {global_idx} out of range")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        batch_idx, local_idx = self._get_batch_and_idx(idx)
        batch_data, batch_labels = self._load_batch(batch_idx)
        
        img_array = batch_data[local_idx]
        # Convert numpy array to PIL Image
        img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        img = self.transform(img)
        label = batch_labels[local_idx]
        return img, label
