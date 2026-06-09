# Oasis Map Orientation Surveyor

地图相对旋转方向检测模型。给定两张地图截图，判断它们之间的相对旋转角度（8分类：0°/45°/90°/135°/180°/225°/270°/315°）。

## 项目结构

```
.
├── config.py                   # 统一配置管理（路径、超参数）
├── dataset.py                  # 数据集定义（RelativeRotationDataset）
├── model.py                    # 模型定义（EfficientNet-B0 / MobileNetV3）
├── train.py                    # 训练脚本（两阶段训练策略）
├── evaluate.py                 # 评估脚本（支持 test/val/real 三种数据集）
├── export.py                   # ONNX 导出
├── inference.py                # ONNX Runtime 推理封装
├── finetune_real.py            # 真实数据微调脚本
├── annotate.html               # 方向标注工具（浏览器打开即可使用）
├── data_utils/                 # 数据生成工具
│   ├── oblique_generator.py    # 卫星图合成数据生成（v3，当前使用）
│   ├── synthetic_generator.py  # 地图瓦片合成数据生成（v1）
│   └── real_data_collector.py  # 真实数据收集
├── dataset/                    # 合成数据集（gitignored）
│   ├── train/                  # 8000张（每类1000）
│   ├── val/                    # 1600张（每类200）
│   └── test/                   # 400张（每类50）独立测试集
├── dataset_raw/                # 未标注的真实截图（gitignored）
├── dataset_real_labeled/       # 已标注的真实数据（gitignored）
├── outputs/                    # 模型输出（gitignored）
│   ├── best_model.pth
│   ├── finetuned_model.pth
│   ├── rotation_model.onnx
│   └── confusion_matrix_*.png
└── venv/                       # Python虚拟环境
```

## 快速开始

### 环境准备

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install opencv-python-headless onnx onnxruntime timm
pip install scikit-learn matplotlib seaborn pillow numpy
```

### 1. 生成合成训练数据

```bash
python data_utils/oblique_generator.py
```

生成约 9600 张卫星图（8000 train + 1600 val），保存到 `dataset/`。

### 2. 训练模型

```bash
python train.py
```

训练策略：
- **Phase A** (Epoch 1-5): 冻结 backbone，仅训练分类头，lr=1e-3，warmup
- **Phase B** (Epoch 6+): 解冻全部参数，lr=1e-4，cosine annealing
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

predictor = RotationPredictor("outputs/rotation_model.onnx", image_size=256)
result = predictor.predict(Image.open("map_screenshot.png"))
print(result)
# {'class': 2, 'angle': 90, 'class_name': '90°', 'confidence': 0.95, ...}
```

## 真实数据闭环（P0）

合成数据上 100% 准确率不代表真实场景表现好。建议按以下流程建立真实数据闭环：

### 收集真实截图

```bash
python data_utils/real_data_collector.py
```

从与训练不同的城市收集约 400 张真实地图截图，保存到 `dataset_raw/`。

### 标注方向

1. 用浏览器打开 `annotate.html`
2. 选择 `dataset_raw/` 文件夹
3. 为每张图片标注旋转方向（快捷键 1-8）
4. 下载 `labels.json`

### 整理标注数据

将标注数据按 8:2 分为 train/val，放入 `dataset_real_labeled/`：

```
dataset_real_labeled/
  train/
    0/ *.png
    1/ *.png
    ...
  val/
    0/ *.png
    ...
```

### 评估基线

```bash
python evaluate.py --dataset real --save-json
```

记录真实数据上的 baseline 指标。

### 微调模型

```bash
python finetune_real.py --pretrained outputs/best_model.pth \
    --data-root dataset_real_labeled \
    --output outputs/finetuned_model.pth
```

微调策略：分层解冻（底层冻结、高层微调），轻量增强，更低学习率。

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
| `TRAIN_CONFIG["batch_size"]` | 32 | 训练批次 |
| `TRAIN_CONFIG["learning_rate"]` | 1e-3 | 初始学习率 |
| `TRAIN_CONFIG["freeze_epochs"]` | 5 | 冻结 backbone 轮数 |

修改 `config.py` 即可全局生效，无需逐个脚本修改。

## 性能指标

### 合成数据（当前模型）

| 数据集 | 样本数 | Overall Acc | Top-2 Acc | Adjacent Acc | Mean Angle Error |
|---|---|---|---|---|---|
| Train | 8000 | 100% | 100% | 100% | 0.0° |
| Val | 1600 | 100% | 100% | 100% | 0.0° |
| **Test** | **400** | **100%** | **100%** | **100%** | **0.0°** |

> ⚠️ 合成数据上的 100% 准确率不代表真实场景表现。务必在真实数据上评估。

### 真实数据（待补充）

收集并标注真实数据后，运行 `evaluate.py --dataset real` 获取指标。

## 技术细节

### 数据增强（训练时）

- ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15)
- GaussianBlur (kernel=5, sigma=0.1-2.0, p=0.5)
- RandomAdjustSharpness (factor=2, p=0.3)
- RandomGrayscale (p=0.05)
- Resize to 256x256
- ImageNet 归一化

### 模型架构

- Backbone: EfficientNet-B0 (ImageNet 预训练)
- 分类头: Linear(1280 → 8)
- 参数量: ~4M

### 训练策略

- 优化器: AdamW
- 学习率调度: Warmup (3 epochs) → Cosine Annealing
- 正则化: Weight decay 1e-4, Label smoothing 0.1, Gradient clipping 1.0
- 早停: patience=10, min_delta=0.001

## 已知问题与改进方向

1. **域差距**: 合成卫星图与真实手机截图存在显著差异，必须通过真实数据微调
2. **角度连续性**: 当前为分类问题，可考虑添加角度回归分支
3. **推理优化**: 可添加 TTA（测试时增强）、多尺度推理
4. **更多 Backbone**: 当前仅 EfficientNet-B0，可尝试 EfficientNet-B2/B3、ConvNeXt

## License

MIT
