"""
Oasis Map Orientation Surveyor - 统一配置管理
所有路径和超参数集中在此，消除硬编码
"""

from pathlib import Path

# =============================================================================
# 项目根目录
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()

# =============================================================================
# 数据集路径
# =============================================================================
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATASET_TRAIN = DATASET_ROOT / "train"
DATASET_VAL = DATASET_ROOT / "val"
DATASET_TEST = DATASET_ROOT / "test"  # 独立测试集

# 真实数据路径（使用C盘避免权限问题）
REAL_DATA_RAW = Path(r"C:\Users\Administrator\dataset_raw")           # 未标注的真实截图
REAL_DATA_LABELED = Path(r"C:\Users\Administrator\dataset_real_labeled")  # 已标注的真实数据
REAL_DATA_TRAIN = REAL_DATA_LABELED / "train"
REAL_DATA_VAL = REAL_DATA_LABELED / "val"

# =============================================================================
# 模型输出路径
# =============================================================================
# 输出目录：优先使用项目目录，无权限时回退到C盘
_PROJECT_OUTPUT = PROJECT_ROOT / "outputs"
_C_USER_OUTPUT = Path(r"C:\Users\Administrator\model_output")
if _PROJECT_OUTPUT.exists() or True:  # 尝试项目目录
    OUTPUT_DIR = _PROJECT_OUTPUT
else:
    OUTPUT_DIR = _C_USER_OUTPUT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 优先使用 outputs/ 下的模型，回退到根目录（兼容旧版本）
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pth" if (OUTPUT_DIR / "best_model.pth").exists() else PROJECT_ROOT / "best_model.pth"
FINETUNED_MODEL_PATH = OUTPUT_DIR / "finetuned_model.pth" if (OUTPUT_DIR / "finetuned_model.pth").exists() else PROJECT_ROOT / "finetuned_model.pth"
ONNX_PATH = OUTPUT_DIR / "rotation_model.onnx"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

# 训练日志
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 数据生成配置
# =============================================================================
# 合成数据生成
SYNTHETIC_IMAGE_SIZE = 512      # 生成时的原始尺寸
SYNTHETIC_CITIES_PER_CLASS = 8  # 每类使用几个城市
SYNTHETIC_TRAIN_PER_CLASS = 1000
SYNTHETIC_VAL_PER_CLASS = 200

# 真实数据收集
REAL_CITIES_COUNT = 8           # 收集真实数据的城市数
REAL_IMAGES_PER_CLASS = 50      # 每类收集多少张
REAL_ZOOM_LEVELS = [15, 16, 17]

# =============================================================================
# 模型配置
# =============================================================================
NUM_CLASSES = 8                 # 8个方向区间 (0°, 45°, ..., 315°)
ANGLE_PER_CLASS = 360 // NUM_CLASSES  # 45°
IMAGE_SIZE = 256                # 模型输入尺寸
BACKBONE = "efficientnet_b0"    # 默认backbone

# =============================================================================
# 训练超参数
# =============================================================================
TRAIN_CONFIG = {
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "max_grad_norm": 1.0,
    "warmup_epochs": 3,
    "freeze_epochs": 5,        # Phase A: 冻结backbone的轮数
    "early_stop_patience": 10,
    "early_stop_min_delta": 0.001,
}

# 微调超参数
FINETUNE_CONFIG = {
    "batch_size": 16,
    "num_epochs": 30,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "label_smoothing": 0.05,
    "early_stop_patience": 10,
}

# =============================================================================
# 数据增强配置
# =============================================================================
TRAIN_AUGMENTATION = {
    "color_jitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.15},
    "gaussian_blur": {"kernel_size": 5, "sigma": (0.1, 2.0), "p": 0.5},
    "sharpness": {"factor": 2, "p": 0.3},
    "grayscale": {"p": 0.05},
}

# ImageNet 标准归一化
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# ONNX 导出配置
# =============================================================================
ONNX_CONFIG = {
    "opset_version": 17,
    "do_constant_folding": True,
    "input_names": ["input"],
    "output_names": ["output"],
    "dynamic_axes": {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
}

# =============================================================================
# 推理配置
# =============================================================================
INFERENCE_CONFIG = {
    "image_size": IMAGE_SIZE,   # 必须与训练一致！
    "providers": None,          # None = 自动选择 (CUDA > CPU)
}

# =============================================================================
# 类别定义
# =============================================================================
CLASS_NAMES = [f"{i * ANGLE_PER_CLASS}°" for i in range(NUM_CLASSES)]
CLASS_ANGLES = [i * ANGLE_PER_CLASS for i in range(NUM_CLASSES)]


def print_config():
    """打印当前配置（用于调试）"""
    print("=" * 60)
    print("Project Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"Output Dir:   {OUTPUT_DIR}")
    print(f"Image Size:   {IMAGE_SIZE}")
    print(f"Backbone:     {BACKBONE}")
    print(f"Classes:      {CLASS_NAMES}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
