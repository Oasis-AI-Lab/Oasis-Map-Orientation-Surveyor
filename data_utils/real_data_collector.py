"""
real_data_collector.py
收集真实地图截图并自动标注旋转方向。
策略：下载大图后，用已知的旋转角度生成带标注的真实样本。

输出目录: dataset_real_labeled/{train,val}/{class_idx}/
"""

import os
import math
import random
import time
import io
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import cv2
from PIL import Image
import requests
from tqdm import tqdm

# 复用 oblique_generator 的下载逻辑
import sys
sys.path.insert(0, str(Path(__file__).parent))
from oblique_generator import (
    download_map_region, rotate_image, GLOBAL_CITIES,
    TILE_SOURCES, REQUEST_TIMEOUT, REQUEST_DELAY, HEADERS, MAX_RETRIES
)

# 配置
# 输出目录：优先项目目录，无权限时回退到C盘
_PROJECT_DIR = Path("e:/github projects/OasisCompany/Oasis-Map-Orientation-Surveyor/dataset_real_labeled")
_C_DIR = Path("C:/Users/Administrator/dataset_real_labeled")
OUTPUT_DIR = _C_DIR  # 使用C盘避免权限问题
TRAIN_RATIO = 0.8
IMAGES_PER_CLASS = 50  # 每类50张，共400张真实样本

# 使用与训练和测试都不同的城市
REAL_CITIES = [
    (41.8781, -87.6298),   # Chicago
    (25.7617, -80.1918),   # Miami
    (43.6532, -79.3832),   # Toronto
    (19.0760, 72.8777),    # Mumbai
    (30.0444, 31.2357),    # Cairo
    (-23.5505, -46.6333),  # Sao Paulo
    (52.5200, 13.4050),    # Berlin
    (40.4168, -3.7038),    # Madrid
]


def collect_real_labeled_data():
    """
    从真实地图下载数据，通过已知旋转创建标注。
    与 eval 的区别：使用 OpenCV 旋转，不加缝隙。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        for class_idx in range(8):
            (OUTPUT_DIR / split / str(class_idx)).mkdir(parents=True, exist_ok=True)

    all_samples = []  # (image, class_idx)

    for city_idx, (lat, lon) in enumerate(REAL_CITIES):
        print(f"Downloading city {city_idx+1}/{len(REAL_CITIES)}: ({lat:.4f}, {lon:.4f})")
        base_map = download_map_region(lat, lon, zoom=17, tiles_x=3, tiles_y=3)
        if base_map is None:
            continue

        # 为每个角度生成样本
        for class_idx in range(8):
            angle = class_idx * 45.0
            rotated = rotate_image(base_map, angle)

            # 随机裁剪多个样本
            samples_per_city = IMAGES_PER_CLASS // len(REAL_CITIES)
            for i in range(samples_per_city):
                crop_size = random.randint(512, min(1024, rotated.size[0], rotated.size[1]))
                x = random.randint(0, max(0, rotated.size[0] - crop_size))
                y = random.randint(0, max(0, rotated.size[1] - crop_size))
                cropped = rotated.crop((x, y, x + crop_size, y + crop_size))
                final = cropped.resize((512, 512), Image.Resampling.BILINEAR)
                all_samples.append((final, class_idx))

        del base_map

    # 随机划分 train/val
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # 保存
    for img, class_idx in train_samples:
        idx = len(list((OUTPUT_DIR / "train" / str(class_idx)).glob("*.png")))
        img.save(OUTPUT_DIR / "train" / str(class_idx) / f"real_{idx:04d}.png")

    for img, class_idx in val_samples:
        idx = len(list((OUTPUT_DIR / "val" / str(class_idx)).glob("*.png")))
        img.save(OUTPUT_DIR / "val" / str(class_idx) / f"real_{idx:04d}.png")

    print(f"\nCollected {len(train_samples)} train + {len(val_samples)} val samples")
    print_summary()


def print_summary():
    print("\n" + "=" * 50)
    print("Real Labeled Dataset Statistics")
    print("=" * 50)
    for split in ["train", "val"]:
        total = 0
        for class_idx in range(8):
            count = len(list((OUTPUT_DIR / split / str(class_idx)).glob("*.png")))
            total += count
            print(f"  {split}/{class_idx} ({class_idx*45}°): {count}")
        print(f"  {split} total: {total}")
    print("=" * 50)


if __name__ == "__main__":
    print("=" * 60)
    print("Oasis-Map-Orientation-Surveyor Real Data Collector")
    print("=" * 60)
    random.seed(456)
    np.random.seed(456)
    collect_real_labeled_data()
