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
    """早停机制：验证损失连续 patience 轮不下降则停止训练。"""

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


def get_linear_warmup_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """创建带 warmup 的余弦退火调度器。"""

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(warmup_epochs)
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)

    return warmup_scheduler, cosine_scheduler


def train_one_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm: float = 1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(all_labels, all_preds, save_path: str):
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    class_names = [f"{i*45}°" for i in range(8)]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Relative Rotation Direction - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

    print("\nConfusion Matrix (Raw Counts):")
    print(cm)


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 训练超参数
    DATASET_ROOT = "dataset"
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 3
    LABEL_SMOOTHING = 0.1
    MAX_GRAD_NORM = 1.0
    EARLY_STOP_PATIENCE = 10

    train_dataset = RelativeRotationDataset(DATASET_ROOT, split="train", image_size=IMAGE_SIZE)
    val_dataset = RelativeRotationDataset(DATASET_ROOT, split="val", image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train class distribution: {train_dataset.get_class_counts()}")
    print(f"Val class distribution: {val_dataset.get_class_counts()}")

    model = build_model(num_classes=8, pretrained=True).to(DEVICE)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    warmup_scheduler, cosine_scheduler = get_linear_warmup_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)

    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=0.001)

    best_val_acc = 0.0
    best_model_path = "best_model.pth"

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # Warmup 阶段使用 warmup_scheduler，之后切换到 cosine_scheduler
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            cosine_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, MAX_GRAD_NORM)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Val Acc: {best_val_acc:.4f})")

        if early_stopping(val_loss):
            print(f"  -> Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\nTraining completed. Best Val Acc: {best_val_acc:.4f}")

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
    _, _, final_preds, final_labels = validate(model, val_loader, criterion, DEVICE)

    print("\n" + "=" * 60)
    print("Final Evaluation on Validation Set")
    print("=" * 60)
    plot_confusion_matrix(final_labels, final_preds, "confusion_matrix.png")

    per_class_acc = []
    for i in range(8):
        mask = final_labels == i
        if mask.sum() > 0:
            acc = (final_preds[mask] == final_labels[mask]).mean()
            per_class_acc.append(acc)
            print(f"Class {i} ({i*45}°) Accuracy: {acc:.4f}")

    print(f"\nMean Class Accuracy: {np.mean(per_class_acc):.4f}")
    print(f"Min Class Accuracy:  {min(per_class_acc):.4f}")
    print(f"Overall Accuracy:    {(final_preds == final_labels).mean():.4f}")

    # Top-2 准确率
    model.eval()
    top2_correct = 0
    top2_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, top2_preds = outputs.topk(2, dim=1)
            top2_correct += top2_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            top2_total += labels.size(0)
    print(f"Top-2 Accuracy:       {top2_correct / top2_total:.4f}")


if __name__ == "__main__":
    main()
