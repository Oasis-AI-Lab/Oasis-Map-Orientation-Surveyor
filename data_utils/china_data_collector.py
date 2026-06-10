"""
china_data_collector.py
专门收集中国地图数据（高德/百度卫星图），增加真实数据多样性。
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from oblique_generator import download_map_region, rotate_image, TILE_SOURCES

# 配置
OUTPUT_DIR = Path("C:/Users/Administrator/dataset_real_labeled_china")
TRAIN_RATIO = 0.8
IMAGES_PER_CLASS = 50  # 每类50张，共400张

# 中国城市（使用高德/百度地图源）
CHINA_CITIES = [
    (39.9042, 116.4074),   # 北京
    (31.2304, 121.4737),   # 上海
    (23.1291, 113.2644),   # 广州
    (30.5728, 104.0668),   # 成都
    (29.5630, 106.5516),   # 重庆
    (34.3416, 108.9398),   # 西安
    (30.2741, 120.1551),   # 杭州
    (24.4798, 118.0894),   # 厦门
    (36.0671, 120.3826),   # 青岛
    (38.9140, 121.6147),   # 大连
]

# 优先使用中国地图源（高德）
CHINA_TILE_SOURCES = [s for s in TILE_SOURCES if "Amap" in s["name"]]


def collect_china_data():
    """收集中国地图数据"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val"]:
        for class_idx in range(8):
            (OUTPUT_DIR / split / str(class_idx)).mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    for city_idx, (lat, lon) in enumerate(CHINA_CITIES):
        print(f"Downloading China city {city_idx+1}/{len(CHINA_CITIES)}: ({lat:.4f}, {lon:.4f})")
        
        # 使用高德地图源
        source = random.choice(CHINA_TILE_SOURCES)
        zoom = random.choice([15, 16, 17])
        
        base_map = download_map_region(lat, lon, zoom=zoom, tiles_x=3, tiles_y=3)
        if base_map is None:
            print(f"  Failed to download, skipping...")
            continue
        
        # 为每个角度生成样本
        for class_idx in range(8):
            angle = class_idx * 45.0
            rotated = rotate_image(base_map, angle)
            
            # 随机裁剪多个样本
            samples_per_city = max(1, IMAGES_PER_CLASS // len(CHINA_CITIES))
            
            for i in range(samples_per_city):
                crop_size = random.randint(512, min(1024, rotated.size[0], rotated.size[1]))
                x = random.randint(0, max(0, rotated.size[0] - crop_size))
                y = random.randint(0, max(0, rotated.size[1] - crop_size))
                cropped = rotated.crop((x, y, x + crop_size, y + crop_size))
                final = cropped.resize((512, 512), Image.Resampling.BILINEAR)
                all_samples.append((final, class_idx))
        
        del base_map
    
    # 划分 train/val
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # 保存
    for img, class_idx in train_samples:
        idx = len(list((OUTPUT_DIR / "train" / str(class_idx)).glob("*.png")))
        img.save(OUTPUT_DIR / "train" / str(class_idx) / f"china_{idx:04d}.png")
    
    for img, class_idx in val_samples:
        idx = len(list((OUTPUT_DIR / "val" / str(class_idx)).glob("*.png")))
        img.save(OUTPUT_DIR / "val" / str(class_idx) / f"china_{idx:04d}.png")
    
    print(f"\nCollected {len(train_samples)} train + {len(val_samples)} val samples from China")
    print_summary()


def print_summary():
    print("\n" + "=" * 50)
    print("China Dataset Statistics")
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
    print("Oasis-Map-Orientation-Surveyor China Data Collector")
    print(f"Target: {IMAGES_PER_CLASS * 8} samples from China cities")
    print("=" * 60)
    random.seed(789)
    np.random.seed(789)
    collect_china_data()
