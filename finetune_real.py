"""
finetune_real.py
使用少量真实标注数据对预训练模型进行微调。
支持：BN 冻结、渐进式解冻、合成+真实混合训练。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import numpy as np
import argparse

from config import (
    DATASET_ROOT, IMAGE_SIZE, NUM_CLASSES, BACKBONE,
    FINETUNE_CONFIG, PROGRESSIVE_UNFREEZE, MIXED_TRAINING,
    IMAGENET_MEAN, IMAGENET_STD, FINETUNED_MODEL_PATH,
)
from dataset import RelativeRotationDataset
from model import build_model


class RealDataDataset(RelativeRotationDataset):
    """
    真实数据数据集。继承 RelativeRotationDataset，但使用较轻的增强。
    假设真实数据目录结构: dataset_real_labeled/{train,val}/{label}/*.png
    """

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 256):
        # 调用父类初始化，但覆盖 transform
        super().__init__(root_dir, split=split, image_size=image_size)

        # 微调时使用较轻的增强，避免破坏真实特征
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def get_class_counts(self):
        """返回 list（与父类一致）"""
        counts = [0] * 8
        for _, label in self.samples:
            counts[label] += 1
        return counts


def freeze_bn(model):
    """冻结所有 BatchNorm 层，使用预训练统计量"""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False


def set_requires_grad_by_block(model, min_block: int):
    """
    按 block 索引设置 requires_grad。
    EfficientNet-B0 features 命名格式: {block_idx}.{sub_idx}.param_name
    """
    for name, param in model.backbone.features.named_parameters():
        parts = name.split('.')
        block_idx = int(parts[0]) if parts[0].isdigit() else -1
        param.requires_grad = (block_idx >= min_block)

    # classifier 始终可训练
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True


def create_real_dataloader(real_data_root: str, image_size: int, batch_size: int):
    """仅使用真实数据创建 DataLoader（纯真实数据微调，避免合成数据稀释）"""
    real_train = RealDataDataset(real_data_root, split="train", image_size=image_size)
    real_val = RealDataDataset(real_data_root, split="val", image_size=image_size)

    train_loader = DataLoader(
        real_train, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        real_val, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, real_train, real_val


def finetune(
    pretrained_path: str,
    real_data_root: str,
    output_path: str = None,
    image_size: int = None,
    batch_size: int = None,
    num_epochs: int = None,
    learning_rate: float = None,
    weight_decay: float = None,
):
    if output_path is None:
        output_path = str(FINETUNED_MODEL_PATH)
    if image_size is None:
        image_size = IMAGE_SIZE
    if batch_size is None:
        batch_size = FINETUNE_CONFIG["batch_size"]
    if num_epochs is None:
        num_epochs = FINETUNE_CONFIG["num_epochs"]
    if learning_rate is None:
        learning_rate = FINETUNE_CONFIG["learning_rate"]
    if weight_decay is None:
        weight_decay = FINETUNE_CONFIG["weight_decay"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练模型
    model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    state_dict = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded pretrained model: {pretrained_path}")

    # 创建真实数据 DataLoader
    train_loader, val_loader, real_train, real_val = create_real_dataloader(
        real_data_root, image_size, batch_size
    )

    print(f"Real train samples: {len(real_train)}")
    print(f"Real val samples: {len(real_val)}")
    print(f"Real class distribution: {real_train.get_class_counts()}")
    print(f"Training batches: {len(train_loader)}")

    if len(real_train) == 0:
        raise ValueError("No real training data found!")

    # 初始状态：冻结所有 backbone，仅训练 classifier
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True

    # 冻结 BN
    freeze_bn(model)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    criterion = nn.CrossEntropyLoss(label_smoothing=FINETUNE_CONFIG["label_smoothing"])

    best_val_acc = 0.0
    patience_counter = 0
    patience = FINETUNE_CONFIG["early_stop_patience"]
    min_delta = 0.001

    for epoch in range(1, num_epochs + 1):
        # ===== 渐进式解冻 =====
        if epoch == PROGRESSIVE_UNFREEZE["phase1_end"] + 1:
            print(f"\nPhase 2 (Epoch {epoch}+): Unfreeze blocks 6-7, lr={PROGRESSIVE_UNFREEZE['phase2_lr']}")
            set_requires_grad_by_block(model, min_block=6)
            freeze_bn(model)  # 重新冻结 BN
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=PROGRESSIVE_UNFREEZE["phase2_lr"], weight_decay=weight_decay
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - epoch + 1, eta_min=1e-7)

        elif epoch == PROGRESSIVE_UNFREEZE["phase2_end"] + 1:
            print(f"\nPhase 3 (Epoch {epoch}+): Unfreeze blocks 5-8, lr={PROGRESSIVE_UNFREEZE['phase3_lr']}")
            set_requires_grad_by_block(model, min_block=5)
            freeze_bn(model)  # 重新冻结 BN
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=PROGRESSIVE_UNFREEZE["phase3_lr"], weight_decay=weight_decay
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - epoch + 1, eta_min=1e-7)

        # ===== 训练 =====
        model.train()
        # 确保 BN 保持 eval 模式
        freeze_bn(model)

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # 仅裁剪可训练参数
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # ===== 验证 =====
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

        # scheduler step 在 optimizer.step 之后
        scheduler.step()

        print(f"Epoch [{epoch}/{num_epochs}] | "
              f"Train: {train_loss/train_total:.4f}/{train_acc:.4f} | "
              f"Val: {avg_val_loss:.4f}/{val_acc:.4f}")

        # 统一使用 val_acc 做模型保存和早停
        if val_acc > best_val_acc + min_delta:
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
    parser.add_argument("--pretrained", type=str, default=str(FINETUNED_MODEL_PATH.parent / "best_model.pth"),
                        help="Path to pretrained model")
    parser.add_argument("--real-data", type=str, default=str(Path.home() / "dataset_real_labeled"),
                        help="Path to labeled real data directory")
    parser.add_argument("--output", type=str, default=str(FINETUNED_MODEL_PATH),
                        help="Output model path")
    parser.add_argument("--epochs", type=int, default=FINETUNE_CONFIG["num_epochs"])
    parser.add_argument("--lr", type=float, default=FINETUNE_CONFIG["learning_rate"])
    parser.add_argument("--batch-size", type=int, default=FINETUNE_CONFIG["batch_size"])

    args = parser.parse_args()

    finetune(
        pretrained_path=args.pretrained,
        real_data_root=args.real_data,
        output_path=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
