"""
evaluate_real.py
用真实测试数据评估模型表现。
加载 ONNX 模型，对 dataset_real_test/ 中的图像进行预测，
输出各类别准确率、混淆矩阵和整体指标。
"""

import sys
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ImageNet 标准归一化参数
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
IMAGE_SIZE = 224
CLASSES = [f"{i*45}°" for i in range(8)]


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - MEAN) / STD
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)


def evaluate(model_path: str, test_dir: str):
    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    test_root = Path(test_dir)
    if not test_root.exists():
        print(f"Test directory not found: {test_dir}")
        print("Please run eval/real_eval_generator.py first.")
        return

    all_preds = []
    all_labels = []
    all_confidences = []

    # 按类别遍历
    for class_idx in range(8):
        class_dir = test_root / str(class_idx)
        if not class_dir.exists():
            print(f"  Class {class_idx} directory not found, skipping")
            continue

        images = list(class_dir.glob("*.png"))
        if not images:
            print(f"  Class {class_idx}: no images found")
            continue

        print(f"  Evaluating class {class_idx} ({class_idx*45}°): {len(images)} images")

        for img_path in tqdm(images, desc=f"Class {class_idx}", leave=False):
            image = Image.open(img_path).convert("RGB")
            input_array = preprocess(image)
            outputs = session.run(None, {"input": input_array})
            logits = outputs[0][0]

            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / exp_logits.sum()

            pred_class = int(np.argmax(probabilities))
            confidence = float(probabilities[pred_class])

            all_preds.append(pred_class)
            all_labels.append(class_idx)
            all_confidences.append(confidence)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    # 计算指标
    print("\n" + "=" * 60)
    print("REAL DATA EVALUATION RESULTS")
    print("=" * 60)

    # 总体准确率
    overall_acc = (all_preds == all_labels).mean()
    print(f"\nOverall Accuracy: {overall_acc:.4f} ({(all_preds == all_labels).sum()}/{len(all_labels)})")

    # 各类别准确率
    print(f"\nPer-Class Accuracy:")
    per_class_acc = []
    for i in range(8):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean()
            avg_conf = all_confidences[mask].mean()
            per_class_acc.append(acc)
            print(f"  Class {i} ({i*45:>3}°): Acc={acc:.4f}, AvgConf={avg_conf:.4f}, Count={mask.sum()}")
        else:
            per_class_acc.append(0.0)
            print(f"  Class {i} ({i*45:>3}°): No samples")

    print(f"\nMean Class Accuracy: {np.mean(per_class_acc):.4f}")
    print(f"Min  Class Accuracy: {min(per_class_acc):.4f}")

    # Top-2 准确率
    print(f"\nMean Confidence: {all_confidences.mean():.4f}")
    print(f"Low Confidence (<0.5) ratio: {(all_confidences < 0.5).mean():.4f}")

    # 混淆矩阵
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    cm = np.zeros((8, 8), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    header = "        " + "  ".join([f"{i*45:>5}°" for i in range(8)])
    print(header)
    for i in range(8):
        row = f"True {i*45:>3}° " + "  ".join([f"{cm[i][j]:>5}" for j in range(8)])
        print(row)

    # 归一化混淆矩阵
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    print(f"\nNormalized Confusion Matrix:")
    header = "        " + "  ".join([f"{i*45:>5}°" for i in range(8)])
    print(header)
    for i in range(8):
        row = f"True {i*45:>3}° " + "  ".join([f"{cm_norm[i][j]:>5.2f}" for j in range(8)])
        print(row)

    # 分析常见误判
    print(f"\nMost Common Misclassifications:")
    misclassifications = []
    for t in range(8):
        for p in range(8):
            if t != p and cm[t][p] > 0:
                misclassifications.append((cm[t][p], t, p))
    misclassifications.sort(reverse=True)
    for count, true_cls, pred_cls in misclassifications[:5]:
        angle_diff = abs(true_cls - pred_cls) * 45
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        print(f"  True {true_cls*45}° -> Pred {pred_cls*45}° (diff={angle_diff}°): {count} times")

    # 保存结果为 JSON
    results = {
        "overall_accuracy": float(overall_acc),
        "mean_class_accuracy": float(np.mean(per_class_acc)),
        "min_class_accuracy": float(min(per_class_acc)),
        "per_class_accuracy": {f"{i*45}deg": float(per_class_acc[i]) for i in range(8)},
        "total_samples": int(len(all_labels)),
        "mean_confidence": float(all_confidences.mean()),
    }

    results_path = Path("eval/real_eval_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    MODEL_PATH = "rotation_model.onnx"
    TEST_DIR = "dataset_real_test"

    if not Path(MODEL_PATH).exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Please run export.py first.")
        sys.exit(1)

    evaluate(MODEL_PATH, TEST_DIR)
