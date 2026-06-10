"""
Oasis Map Orientation Surveyor - 训练脚本
两阶段训练策略：Phase A (冻结backbone) → Phase B (解冻全部)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

from config import (
    DATASET_ROOT, OUTPUT_DIR, BEST_MODEL_PATH, IMAGE_SIZE, NUM_CLASSES,
    BACKBONE, TRAIN_CONFIG, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES,
    WARMUP_EPOCHS, FREEZE_EPOCHS, MIXUP_ALPHA, USE_CSL, CSL_SIGMA,
    CONFUSION_MATRIX_PATH,
)
from dataset import RelativeRotationDataset
from model import build_model, count_parameters


class EarlyStopping:
    """基于验证准确率的早停机制"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc: float) -> bool:
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def mixup_data(x, y, alpha=0.2):
    """Mixup 数据增强：线性插值创建虚拟训练样本"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam





def csl_loss(outputs, labels, num_classes=8, sigma=1.0):
    """
    Circular Smooth Label loss.
    将硬标签转换为环形平滑软标签，使用 CrossEntropyLoss 计算。
    相比 KLDivLoss，不需要 log_softmax，实现更简单稳定。
    """
    batch_size = labels.size(0)
    smooth_labels = torch.zeros(batch_size, num_classes, device=labels.device)
    for i in range(batch_size):
        for j in range(num_classes):
            angle_diff = abs(j - labels[i].item())
            angle_diff = min(angle_diff, num_classes - angle_diff)
            smooth_labels[i, j] = math.exp(-angle_diff ** 2 / (2 * sigma ** 2))
        smooth_labels[i] /= smooth_labels[i].sum()
    # 使用 -sum(p * log_softmax(q)) 形式，等价于 KLDivLoss 但更安全
    log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -(smooth_labels * log_probs).sum(dim=1).mean()
    return loss


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    max_grad_norm: float = 1.0, use_mixup: bool = False,
                    mixup_alpha: float = 0.2, use_csl: bool = False,
                    csl_sigma: float = 1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            optimizer.zero_grad()
            outputs = model(images)
            if use_csl:
                loss_a = csl_loss(outputs, labels_a, NUM_CLASSES, csl_sigma)
                loss_b = csl_loss(outputs, labels_b, NUM_CLASSES, csl_sigma)
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            # Mixup 准确率：取两个标签中概率更高的
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item() +
                       (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            optimizer.zero_grad()
            outputs = model(images)
            if use_csl:
                loss = csl_loss(outputs, labels, NUM_CLASSES, csl_sigma)
            else:
                loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        loss.backward()
        # 仅对可训练参数做梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm
        )
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(all_labels, all_preds, save_path: str):
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # 从 config.py 导入所有配置
    BATCH_SIZE = TRAIN_CONFIG["batch_size"]
    NUM_EPOCHS = TRAIN_CONFIG["num_epochs"]
    LEARNING_RATE = TRAIN_CONFIG["learning_rate"]
    WEIGHT_DECAY = TRAIN_CONFIG["weight_decay"]
    LABEL_SMOOTHING = TRAIN_CONFIG["label_smoothing"]
    MAX_GRAD_NORM = TRAIN_CONFIG["max_grad_norm"]

    train_dataset = RelativeRotationDataset(str(DATASET_ROOT), split="train", image_size=IMAGE_SIZE)
    val_dataset = RelativeRotationDataset(str(DATASET_ROOT), split="val", image_size=IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Train class dist: {train_dataset.get_class_counts()}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Best Model Path: {BEST_MODEL_PATH}")

    model = build_model(num_classes=NUM_CLASSES, pretrained=True, backbone=BACKBONE).to(DEVICE)
    print(f"Model: {BACKBONE}, Parameters: {count_parameters(model):,}")

    # 损失函数：CSL 使用自定义 csl_loss，否则使用 CrossEntropyLoss
    if USE_CSL:
        criterion = None  # CSL 使用 csl_loss 函数，不需要 criterion
        print(f"Loss: CSL (Circular Smooth Label, sigma={CSL_SIGMA})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        print(f"Loss: CrossEntropyLoss (label_smoothing={LABEL_SMOOTHING})")

    # Phase A: 冻结 backbone features，仅训练 classifier
    print(f"\nPhase A (Epoch 1-{FREEZE_EPOCHS}): Frozen backbone, lr={LEARNING_RATE}")
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 修复 H4: 使用 min() 防止 warmup lambda 溢出
    warmup_scheduler = LambdaLR(optimizer, lambda e: min((e + 1) / WARMUP_EPOCHS, 1.0))
    early_stopping = EarlyStopping(
        patience=TRAIN_CONFIG["early_stop_patience"],
        min_delta=TRAIN_CONFIG["early_stop_min_delta"]
    )
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        use_mixup = (epoch > FREEZE_EPOCHS)  # Mixup 仅在 Phase B 启用

        # Phase A→B 切换：解冻 backbone，降低学习率
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\nPhase B (Epoch {epoch}-{NUM_EPOCHS}): Unfrozen backbone, lr={LEARNING_RATE * 0.1}")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=WEIGHT_DECAY)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - FREEZE_EPOCHS, eta_min=1e-6)

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, MAX_GRAD_NORM,
            use_mixup=use_mixup, mixup_alpha=MIXUP_ALPHA,
            use_csl=USE_CSL, csl_sigma=CSL_SIGMA,
        )

        # 验证（使用标准 CrossEntropyLoss 计算验证 loss，不受 CSL 影响）
        val_criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, val_criterion, DEVICE)

        # 修复 H2: scheduler.step() 在 optimizer.step() 之后调用
        if epoch <= FREEZE_EPOCHS:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        mixup_tag = " [Mixup]" if use_mixup else ""
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] LR: {current_lr:.6f} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f}{mixup_tag} | Val: {val_loss:.4f}/{val_acc:.4f} | "
              f"Time: {time.time()-start_time:.1f}s")

        # 修复 M9: 统一使用 val_acc 做模型保存和早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(BEST_MODEL_PATH))
            print(f"  -> Best model saved (Val Acc: {best_val_acc:.4f})")

        if early_stopping(val_acc):
            print(f"  -> Early stopping at epoch {epoch}")
            break

    # 最终评估
    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(str(BEST_MODEL_PATH), map_location=DEVICE, weights_only=True))
    val_criterion = nn.CrossEntropyLoss()
    _, _, final_preds, final_labels = validate(model, val_loader, val_criterion, DEVICE)

    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    plot_confusion_matrix(final_labels, final_preds, str(CONFUSION_MATRIX_PATH))

    per_class_acc = []
    for i in range(NUM_CLASSES):
        mask = final_labels == i
        if mask.sum() > 0:
            acc = (final_preds[mask] == final_labels[mask]).mean()
            per_class_acc.append(acc)
            print(f"  Class {i} ({CLASS_NAMES[i]}): {acc:.4f}")

    print(f"\nMean Class Acc: {np.mean(per_class_acc):.4f}")
    print(f"Min  Class Acc: {min(per_class_acc):.4f}")
    print(f"Overall Acc:    {(final_preds == final_labels).mean():.4f}")

    # Top-2 准确率
    model.eval()
    top2_correct, top2_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, top2 = model(images).topk(2, dim=1)
            top2_correct += top2.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            top2_total += labels.size(0)
    print(f"Top-2 Acc:       {top2_correct / top2_total:.4f}")


if __name__ == "__main__":
    main()
