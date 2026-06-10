# Oasis Map Orientation Surveyor

地图相对旋转方向检测模型。给定两张地图截图，判断它们之间的相对旋转角度（8分类：0°/45°/90°/135°/180°/225°/270°/315°）。

## 项目结构

```
.
├── config.py                   # 统一配置管理（路径、超参数、增强策略）
├── dataset.py                  # 数据集定义（含几何增强）
├── model.py                    # 模型定义（EfficientNet-B0 + Dropout）
├── train.py                    # 训练脚本（两阶段 + Mixup + CSL）
├── evaluate.py                 # 评估脚本（支持 test/val/real）
├── export.py                   # ONNX 导出
├── inference.py                # ONNX Runtime 推理封装
├── finetune_real.py            # 真实数据微调（BN冻结 + 渐进式解冻 + 混合训练）
├── annotate.html               # 方向标注工具
├── requirements.txt            # 依赖管理
├── data_utils/                 # 数据生成工具
│   ├── __init__.py
│   ├── oblique_generator.py    # 卫星图合成数据生成（v3）
│   └── real_data_collector.py  # 真实数据收集
├── dataset/                    # 合成数据集（gitignored）
│   ├── train/                  # 8000张（每类1000）
│   ├── val/                    # 1200张（每类150）
│   └── test/                   # 400张（每类50）独立测试集
├── dataset_raw/                # 未标注的真实截图（gitignored）
├── dataset_real_labeled/       # 已标注的真实数据（gitignored）
└── outputs/                    # 模型输出（gitignored）
```

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

### 1. 生成合成训练数据

```bash
python data_utils/oblique_generator.py
```

生成约 9600 张卫星图（8000 train + 1600 val），保存到 `dataset/`。

### 2. 训练模型

```bash
python -u train.py
```

训练策略：
- **Phase A** (Epoch 1-5): 冻结 backbone，仅训练分类头，lr=1e-3，warmup
- **Phase B** (Epoch 6+): 解冻全部参数，lr=1e-4，cosine annealing，启用 Mixup
- CSL (Circular Smooth Label): 编码角度连续性，相邻类别惩罚更小
- Early stopping (patience=10)

模型保存到 `outputs/best_model.pth`。

### 3. 评估模型

```bash
# 在独立测试集上评估（推荐）
python evaluate.py --dataset test --save-json

# 在验证集上评估
python evaluate.py --dataset val --save-json

# 在真实数据上评估（需先收集并标注）
python evaluate.py --dataset real --save-json
```

输出指标：Overall Accuracy、Top-2 Accuracy、Adjacent Accuracy (±45°)、Mean Angle Error、Per-Class Accuracy。

### 4. 导出 ONNX

```bash
python export.py
```

导出到 `outputs/rotation_model.onnx`（opset 17，动态 batch）。

### 5. 推理

```python
from inference import RotationPredictor
from PIL import Image

predictor = RotationPredictor("outputs/rotation_model.onnx")
result = predictor.predict(Image.open("map_screenshot.png"))
print(result)
# {'class': 2, 'angle': 90, 'class_name': '90°', 'confidence': 0.95, ...}
```

## 真实数据闭环

合成数据上 100% 准确率不代表真实场景表现好。建议按以下流程建立真实数据闭环：

### 收集真实截图

```bash
python data_utils/real_data_collector.py
```

### 标注方向

1. 用浏览器打开 `annotate.html`
2. 选择 `dataset_raw/` 文件夹
3. 为每张图片标注旋转方向（快捷键 1-8）
4. 下载 `labels.json`

### 整理标注数据

将标注数据按 8:2 分为 train/val，放入 `dataset_real_labeled/`。

### 评估基线

```bash
python evaluate.py --dataset real --save-json
```

### 微调模型

```bash
python finetune_real.py --pretrained outputs/best_model.pth \
    --real-data dataset_real_labeled \
    --output outputs/finetuned_model.pth
```

微调策略：
- BN 冻结（使用预训练统计量）
- 渐进式解冻（Phase 1: classifier → Phase 2: blocks 6-7 → Phase 3: blocks 5-8）
- 合成+真实混合训练（真实权重 1.0，合成权重 0.3）
- 更低学习率（1e-4 → 5e-5 → 1e-5）

### 对比效果

```bash
# 微调前
python evaluate.py --model outputs/best_model.pth --dataset real

# 微调后
python evaluate.py --model outputs/finetuned_model.pth --dataset real
```

## 配置说明

所有路径和超参数集中在 `config.py`，关键配置：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `IMAGE_SIZE` | 256 | 模型输入尺寸 |
| `BACKBONE` | efficientnet_b0 | 主干网络 |
| `NUM_CLASSES` | 8 | 方向类别数 |
| `DROPOUT_P` | 0.3 | 分类头前 Dropout 概率 |
| `MIXUP_ALPHA` | 0.2 | Mixup 增强强度 |
| `USE_CSL` | True | 启用 Circular Smooth Label |
| `CSL_SIGMA` | 1.0 | CSL 高斯平滑 sigma |
| `TRAIN_CONFIG["batch_size"]` | 32 | 训练批次 |
| `TRAIN_CONFIG["learning_rate"]` | 1e-3 | 初始学习率 |
| `TRAIN_CONFIG["freeze_epochs"]` | 5 | 冻结 backbone 轮数 |

修改 `config.py` 即可全局生效，无需逐个脚本修改。

## 性能指标

### 合成数据

| 数据集 | 样本数 | Overall Acc | Top-2 Acc | Adjacent Acc | Mean Angle Error |
|---|---|---|---|---|---|
| Train | 8000 | ~99% | 100% | 100% | <1° |
| Val | 1200 | ~99% | 100% | 100% | <1° |
| **Test** | **400** | **~99%** | **100%** | **100%** | **<1°** |

### 真实数据

| 阶段 | Overall Acc | Mean Angle Error |
|---|---|---|
| 微调前（基线） | ~15% | ~90° |
| 微调后 | ≥90%（目标 ≥95%） | <20° |

> ⚠️ 合成数据上的高准确率不代表真实场景表现。务必在真实数据上评估和微调。

## 技术细节

### 数据增强（训练时）

- Resize to 288x288 → RandomCrop to 256x256
- ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15)
- RandomRotation (±10°)
- RandomAffine (translate=5%, shear=2°)
- GaussianBlur (kernel=5, sigma=0.1-2.0, p=0.5)
- RandomAdjustSharpness (factor=2, p=0.3)
- RandomGrayscale (p=0.05)
- ImageNet 归一化

### 模型架构

- Backbone: EfficientNet-B0 (ImageNet 预训练)
- 分类头: Dropout(0.3) → Linear(1280 → 8)
- 参数量: ~4M

### 训练策略

- 优化器: AdamW
- 学习率调度: Warmup (3 epochs) → Cosine Annealing
- 正则化: Weight decay 1e-4, Label smoothing 0.1, Gradient clipping 1.0
- Mixup: alpha=0.2（仅 Phase B 启用）
- CSL: sigma=1.0（编码角度连续性）
- 早停: patience=10, min_delta=0.001

## License

MIT
