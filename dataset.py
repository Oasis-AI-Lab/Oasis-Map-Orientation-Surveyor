import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path


class RelativeRotationDataset(Dataset):
    """
    相对旋转方向数据集
    数据集结构: dataset/{train,val}/{label}/*.png
    标签 0-7 代表 8 个相对旋转区间 (每个区间 45 度)
    """

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 224):
        self.root_dir = Path(root_dir) / split
        self.image_size = image_size
        self.split = split

        self.samples = []
        for label_dir in self.root_dir.iterdir():
            if label_dir.is_dir() and label_dir.name.isdigit():
                label = int(label_dir.name)
                for img_path in label_dir.glob("*.png"):
                    self.samples.append((str(img_path), label))

        self.samples.sort(key=lambda x: (x[1], x[0]))

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

    def get_class_counts(self):
        counts = [0] * 8
        for _, label in self.samples:
            counts[label] += 1
        return counts
