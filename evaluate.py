"""
模型评估脚本
支持在合成测试集、验证集或真实数据上评估模型性能
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    BEST_MODEL_PATH, FINETUNED_MODEL_PATH, IMAGE_SIZE, NUM_CLASSES,
    CLASS_NAMES, ANGLE_PER_CLASS, IMAGENET_MEAN, IMAGENET_STD,
    DATASET_TEST, DATASET_VAL, REAL_DATA_VAL, OUTPUT_DIR
)
from dataset import RelativeRotationDataset
from model import build_model


def evaluate_model(model, dataloader, device, dataset_name="Dataset"):
    """在指定数据集上评估模型，返回详细指标"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels, all_logits = [], [], []
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    # 基础指标
    overall_acc = correct / total
    mean_loss = total_loss / total

    # 每类准确率
    per_class_acc = []
    for i in range(NUM_CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean()
            per_class_acc.append(acc)

    # Top-2 准确率
    top2_correct = 0
    for i in range(len(all_labels)):
        top2 = np.argsort(all_logits[i])[-2:]
        if all_labels[i] in top2:
            top2_correct += 1
    top2_acc = top2_correct / len(all_labels)

    # 角度误差（考虑环形连续性）
    angle_errors = []
    for pred, true in zip(all_preds, all_labels):
        pred_angle = pred * ANGLE_PER_CLASS
        true_angle = true * ANGLE_PER_CLASS
        error = abs(pred_angle - true_angle)
        error = min(error, 360 - error)  # 环形距离
        angle_errors.append(error)
    angle_errors = np.array(angle_errors)

    # 相邻类别准确率（±1类内算正确，用于评估"大致正确"）
    adjacent_correct = 0
    for pred, true in zip(all_preds, all_labels):
        diff = abs(pred - true)
        diff = min(diff, NUM_CLASSES - diff)  # 环形
        if diff <= 1:
            adjacent_correct += 1
    adjacent_acc = adjacent_correct / len(all_labels)

    results = {
        "dataset": dataset_name,
        "num_samples": total,
        "overall_accuracy": float(overall_acc),
        "mean_loss": float(mean_loss),
        "top2_accuracy": float(top2_acc),
        "adjacent_accuracy": float(adjacent_acc),  # ±45°内正确
        "mean_per_class_accuracy": float(np.mean(per_class_acc)),
        "min_per_class_accuracy": float(np.min(per_class_acc)),
        "mean_angle_error_deg": float(np.mean(angle_errors)),
        "median_angle_error_deg": float(np.median(angle_errors)),
        "max_angle_error_deg": float(np.max(angle_errors)),
        "per_class_accuracy": {CLASS_NAMES[i]: float(per_class_acc[i]) for i in range(NUM_CLASSES)},
    }

    return results, all_labels, all_preds


def plot_confusion_matrix(all_labels, all_preds, save_path, title="Confusion Matrix"):
    """绘制混淆矩阵"""
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0, vmax=1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def print_results(results):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print(f"Evaluation Results: {results['dataset']}")
    print("=" * 60)
    print(f"Samples:           {results['num_samples']}")
    print(f"Overall Accuracy:  {results['overall_accuracy']:.4f}")
    print(f"Top-2 Accuracy:    {results['top2_accuracy']:.4f}")
    print(f"Adjacent Acc (±45°): {results['adjacent_accuracy']:.4f}")
    print(f"Mean Class Acc:    {results['mean_per_class_accuracy']:.4f}")
    print(f"Min Class Acc:     {results['min_per_class_accuracy']:.4f}")
    print(f"Mean Loss:         {results['mean_loss']:.4f}")
    print(f"Mean Angle Error:  {results['mean_angle_error_deg']:.1f}°")
    print(f"Median Angle Error: {results['median_angle_error_deg']:.1f}°")
    print(f"Max Angle Error:   {results['max_angle_error_deg']:.1f}°")
    print("-" * 60)
    print("Per-Class Accuracy:")
    for name, acc in results['per_class_accuracy'].items():
        bar = "█" * int(acc * 20)
        print(f"  {name:>6}: {acc:.4f} {bar}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate rotation model")
    parser.add_argument("--model", type=str, default=str(BEST_MODEL_PATH),
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, choices=["test", "val", "real"], default="test",
                        help="Dataset to evaluate on")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--save-json", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 选择数据集
    if args.dataset == "test":
        data_root = str(DATASET_TEST.parent)
        split = "test"
        dataset_name = "Synthetic Test"
    elif args.dataset == "val":
        data_root = str(DATASET_VAL.parent)
        split = "val"
        dataset_name = "Synthetic Validation"
    elif args.dataset == "real":
        if not REAL_DATA_VAL.exists():
            print(f"Error: Real data not found at {REAL_DATA_VAL}")
            print("Please run real data collection and annotation first.")
            return
        data_root = str(REAL_DATA_VAL.parent)
        split = "val"
        dataset_name = "Real Data Validation"

    dataset = RelativeRotationDataset(data_root, split=split, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Dataset: {dataset_name} ({len(dataset)} samples)")

    # 加载模型
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Model loaded: {model_path}")

    # 评估
    results, all_labels, all_preds = evaluate_model(model, dataloader, device, dataset_name)
    print_results(results)

    # 保存混淆矩阵
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cm_path = output_dir / f"confusion_matrix_{args.dataset}.png"
    plot_confusion_matrix(all_labels, all_preds, cm_path, title=f"{dataset_name} - Confusion Matrix")

    # 保存JSON
    if args.save_json:
        json_path = output_dir / f"evaluation_{args.dataset}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved: {json_path}")


if __name__ == "__main__":
    main()
