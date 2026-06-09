"""
finetune_real.py
使用少量真实标注数据对预训练模型进行微调。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import argparse

from dataset import RelativeRotationDataset
from model import build_model


class RealDataDataset(RelativeRotationDataset):
    """
    继承 RelativeRotationDataset，但使用真实数据路径。
    假设真实数据目录结构: dataset_real_labeled/{train,val}/{label}/*.png
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

        # 微调时使用较轻的增强，避免破坏真实特征
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    def get_class_counts(self):
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts


def finetune(
    pretrained_path: str,
    real_data_root: str,
    output_path: str = "finetuned_model.pth",
    image_size: int = 256,
    batch_size: int = 16,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练模型
    model = build_model(num_classes=8, pretrained=False).to(DEVICE)
    state_dict = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded pretrained model: {pretrained_path}")

    # 创建数据集
    train_dataset = RealDataDataset(real_data_root, split="train", image_size=image_size)
    val_dataset = RealDataDataset(real_data_root, split="val", image_size=image_size)

    print(f"Real train samples: {len(train_dataset)}")
    print(f"Real val samples: {len(val_dataset)}")
    print(f"Class distribution: {train_dataset.get_class_counts()}")

    if len(train_dataset) == 0:
        raise ValueError("No real training data found!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 分层解冻策略：
    # - 冻结 backbone 底层 (features 0-4)
    # - 微调 backbone 高层 (features 5-8)
    # - 完全训练 classifier
    for name, param in model.backbone.features.named_parameters():
        block_idx = int(name.split('.')[0]) if name[0].isdigit() else -1
        if block_idx < 5:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # classifier 始终训练
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    min_delta = 0.001
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total

        print(f"Epoch [{epoch}/{num_epochs}] | "
              f"Train: {train_loss/train_total:.4f}/{train_acc:.4f} | "
              f"Val: {avg_val_loss:.4f}/{val_acc:.4f}")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
            print(f"  -> Best model saved (Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nFinetuning complete. Best Val Acc: {best_val_acc:.4f}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune on real data")
    parser.add_argument("--pretrained", type=str, default="best_model.pth",
                        help="Path to pretrained model")
    parser.add_argument("--real-data", type=str, default="dataset_real_labeled",
                        help="Path to labeled real data directory")
    parser.add_argument("--output", type=str, default="finetuned_model.pth",
                        help="Output model path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()

    finetune(
        pretrained_path=args.pretrained,
        real_data_root=args.real_data,
        output_path=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
