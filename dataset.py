import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

from config import IMAGENET_MEAN, IMAGENET_STD, TRAIN_AUGMENTATION


class RelativeRotationDataset(Dataset):
    """
    相对旋转方向数据集
    数据集结构: dataset/{train,val}/{label}/*.png
    标签 0-7 代表 8 个相对旋转区间 (每个区间 45 度)
    """

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 256):
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
            aug = TRAIN_AUGMENTATION
            self.transform = transforms.Compose([
                transforms.Resize((image_size + aug["random_crop_pad"], image_size + aug["random_crop_pad"])),
                transforms.RandomCrop(image_size),
                transforms.ColorJitter(**aug["color_jitter"]),
                transforms.RandomRotation(degrees=aug["random_rotation"]),
                transforms.RandomAffine(
                    degrees=0,
                    translate=aug["random_affine_translate"],
                    shear=aug["random_affine_shear"]
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(**{k: v for k, v in aug["gaussian_blur"].items() if k != "p"})
                ], p=aug["gaussian_blur"]["p"]),
                transforms.RandomApply([
                    transforms.RandomAdjustSharpness(sharpness_factor=aug["sharpness"]["factor"])
                ], p=aug["sharpness"]["p"]),
                transforms.RandomGrayscale(p=aug["grayscale"]["p"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
