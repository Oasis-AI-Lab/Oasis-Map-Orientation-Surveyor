import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path

from dataset import RelativeRotationDataset
from model import build_model, count_parameters


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_one_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm: float = 1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
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
    class_names = [f"{i*45}°" for i in range(8)]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
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

    # 路径
    PROJECT_DIR = Path(__file__).parent
    DATASET_ROOT = str(PROJECT_DIR / "dataset")
    OUTPUT_DIR = Path(r"C:\Users\Administrator\model_output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH = str(OUTPUT_DIR / "best_model.pth")
    CM_PATH = str(OUTPUT_DIR / "confusion_matrix.png")

    # 超参数
    DATASET_ROOT = DATASET_ROOT
    IMAGE_SIZE = 256          # 从 224 提升到 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1
    MAX_GRAD_NORM = 1.0
    WARMUP_EPOCHS = 3
    FREEZE_EPOCHS = 5         # 前 5 轮冻结 backbone
    BACKBONE = "efficientnet_b0"

    train_dataset = RelativeRotationDataset(DATASET_ROOT, split="train", image_size=IMAGE_SIZE)
    val_dataset = RelativeRotationDataset(DATASET_ROOT, split="val", image_size=IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Train class dist: {train_dataset.get_class_counts()}")

    model = build_model(num_classes=8, pretrained=True, backbone=BACKBONE).to(DEVICE)
    print(f"Model: {BACKBONE}, Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Phase A: 冻结 backbone features，仅训练 classifier
    print(f"\nPhase A (Epoch 1-{FREEZE_EPOCHS}): Frozen backbone, lr={LEARNING_RATE}")
    for param in model.backbone.parameters():
        param.requires_grad = False
    # 解冻 classifier（在 backbone 内部）
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: (e + 1) / WARMUP_EPOCHS)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        # Phase A→B 切换：解冻 backbone，降低学习率
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\nPhase B (Epoch {epoch}-{NUM_EPOCHS}): Unfrozen backbone, lr={LEARNING_RATE * 0.1}")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=WEIGHT_DECAY)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - FREEZE_EPOCHS, eta_min=1e-6)

        # 学习率调度
        if epoch <= FREEZE_EPOCHS:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, MAX_GRAD_NORM)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] LR: {current_lr:.6f} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f} | "
              f"Time: {time.time()-start_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Best model saved (Val Acc: {best_val_acc:.4f})")

        if early_stopping(val_loss):
            print(f"  -> Early stopping at epoch {epoch}")
            break

    # 最终评估
    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
    _, _, final_preds, final_labels = validate(model, val_loader, criterion, DEVICE)

    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    plot_confusion_matrix(final_labels, final_preds, CM_PATH)

    per_class_acc = []
    for i in range(8):
        mask = final_labels == i
        if mask.sum() > 0:
            acc = (final_preds[mask] == final_labels[mask]).mean()
            per_class_acc.append(acc)
            print(f"  Class {i} ({i*45}°): {acc:.4f}")

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
