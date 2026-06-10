"""
inference.py
Oasis-Map-Orientation-Surveyor 推理封装模块。
提供 RotationPredictor 类，输入一张地图截图，输出相对旋转方向预测结果。
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Dict, List, Optional

from config import IMAGE_SIZE, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD


class RotationPredictor:
    """
    相对旋转方向预测器。
    使用 ONNX Runtime 进行推理，输入 PIL Image，返回预测结果字典。
    """

    def __init__(self, model_path: str, image_size: int = IMAGE_SIZE, providers: Optional[List[str]] = None):
        """
        Args:
            model_path: ONNX 模型文件路径
            image_size: 模型输入尺寸（正方形），默认从 config.py 导入
            providers: ONNX Runtime 执行提供者，默认自动选择 (CUDA > CPU)
        """
        if providers is None:
            providers = ort.get_available_providers()

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.image_size = image_size
        self.classes = CLASS_NAMES

        # ImageNet 标准归一化参数（从 config.py 导入）
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        预处理图像：Resize -> ToArray -> Normalize -> AddBatchDim -> CHW格式
        """
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        # To numpy array (H, W, C) float32 [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Normalize
        img_array = (img_array - self.mean) / self.std

        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.astype(np.float32)

    def predict(self, image: Image.Image) -> Dict:
        """
        预测单张图像的相对旋转方向。

        Args:
            image: PIL Image 对象

        Returns:
            dict: {
                "class": int,          # 预测类别 0-7
                "angle": int,          # 预测角度 0, 45, 90, ..., 315
                "class_name": str,     # 类别名 "0°", "45°", ...
                "confidence": float,    # 预测置信度 0.0-1.0
                "probabilities": dict  # 各类别概率 {class_name: probability}
            }
        """
        input_array = self.preprocess(image)

        # ONNX 推理
        outputs = self.session.run(None, {"input": input_array})
        logits = outputs[0][0]  # (8,)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        # 预测结果
        pred_class = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_class])

        return {
            "class": pred_class,
            "angle": pred_class * 45,
            "class_name": self.classes[pred_class],
            "confidence": confidence,
            "probabilities": {self.classes[i]: float(probabilities[i]) for i in range(8)}
        }

    def predict_batch(self, images: List[Image.Image]) -> List[Dict]:
        """
        批量预测多张图像。

        Args:
            images: PIL Image 列表

        Returns:
            List[Dict]: 预测结果列表
        """
        if not images:
            return []

        # 批量预处理
        batch_array = np.concatenate([self.preprocess(img) for img in images], axis=0)

        # ONNX 批量推理
        outputs = self.session.run(None, {"input": batch_array})
        logits = outputs[0]  # (N, 8)

        results = []
        for i in range(len(images)):
            exp_logits = np.exp(logits[i] - np.max(logits[i]))
            probabilities = exp_logits / exp_logits.sum()

            pred_class = int(np.argmax(probabilities))
            confidence = float(probabilities[pred_class])

            results.append({
                "class": pred_class,
                "angle": pred_class * 45,
                "class_name": self.classes[pred_class],
                "confidence": confidence,
                "probabilities": {self.classes[j]: float(probabilities[j]) for j in range(8)}
            })

        return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from config import ONNX_PATH
    model_path = str(ONNX_PATH)
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Please run export.py first.")
        sys.exit(1)

    # 测试推理
    predictor = RotationPredictor(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Input size: {predictor.image_size}")
    print(f"Classes: {predictor.classes}")

    # 创建随机测试图像
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    result = predictor.predict(test_image)

    print(f"\nPrediction result:")
    print(f"  Class: {result['class']} ({result['class_name']})")
    print(f"  Angle: {result['angle']}°")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Probabilities:")
    for cls_name, prob in result["probabilities"].items():
        bar = "█" * int(prob * 40)
        print(f"    {cls_name}: {prob:.4f} {bar}")
