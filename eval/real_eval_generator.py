"""
real_eval_generator.py
从在线地图服务下载真实卫星图，生成带已知旋转方向的测试集。
用于评估模型在真实数据上的表现。

v3 优化：
- 去掉瓦片缝隙模拟（真实截图不会有瓦片边界）
- 改用 OpenCV 旋转（与训练一致）
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
from PIL import Image, ImageDraw
import requests
from tqdm import tqdm


# ==================== 配置参数 ====================

# 多地图源配置（Esri 优先，Google 在中国可能超时）
TILE_SOURCES = [
    {
        "name": "Esri World Imagery",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "type": "satellite",
    },
    {
        "name": "Bing Aerial",
        "url": "https://t0.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1",
        "type": "satellite",
        "use_quadkey": True,
    },
    {
        "name": "Google Satellite",
        "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "type": "satellite",
    },
]

TILE_SIZE = 256
MAX_ZOOM = 19
MIN_ZOOM = 16

# 测试集配置
TARGET_CLASSES = 8
IMAGES_PER_CLASS_TEST = 50  # 每个方向 50 张测试图
CROP_SIZE_MIN = 512
CROP_SIZE_MAX = 1024
FINAL_SAVE_SIZE = 512

# 输出目录
OUTPUT_DIR = Path("e:/github projects/OasisCompany/Oasis-Map-Orientation-Surveyor/dataset_real_test")

# 请求配置
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 0.1
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
MAX_RETRIES = 3


# ==================== 坐标转换工具 ====================

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_xy_to_quadkey(x: int, y: int, z: int) -> str:
    quadkey = ""
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey


# ==================== 地图切片下载 ====================

def get_tile_url(source_idx: int, x: int, y: int, z: int) -> str:
    source = TILE_SOURCES[source_idx % len(TILE_SOURCES)]
    url = source["url"]
    if source.get("use_quadkey"):
        quadkey = tile_xy_to_quadkey(x, y, z)
        url = url.replace("{quadkey}", quadkey)
    else:
        url = url.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
    return url


def download_tile(x: int, y: int, z: int, source_idx: int = 0, retries: int = MAX_RETRIES) -> Optional[Image.Image]:
    for attempt in range(retries):
        try:
            current_source_idx = (source_idx + attempt) % len(TILE_SOURCES)
            url = get_tile_url(current_source_idx, x, y, z)
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                if attempt < retries - 1:
                    continue
                return None
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            time.sleep(REQUEST_DELAY)
            return image
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                print(f"Download failed ({x}, {y}, {z}): {e}")
                return None
    return None


def download_map_region(
    center_lat: float,
    center_lon: float,
    zoom: int,
    tiles_x: int = 3,
    tiles_y: int = 3,
    source_idx: int = 0
) -> Optional[Image.Image]:
    center_x, center_y = deg2num(center_lat, center_lon, zoom)
    start_x = center_x - tiles_x // 2
    start_y = center_y - tiles_y // 2
    total_width = tiles_x * TILE_SIZE
    total_height = tiles_y * TILE_SIZE
    big_image = Image.new("RGB", (total_width, total_height), color=(30, 30, 30))

    source_name = TILE_SOURCES[source_idx % len(TILE_SOURCES)]["name"]
    print(f"Downloading: center ({center_lat:.4f}, {center_lon:.4f}), Zoom={zoom}, source={source_name}")

    success_count = 0
    for dy in range(tiles_y):
        for dx in range(tiles_x):
            tile_x = start_x + dx
            tile_y = start_y + dy
            tile = download_tile(tile_x, tile_y, zoom, source_idx=source_idx)
            if tile is not None:
                big_image.paste(tile, (dx * TILE_SIZE, dy * TILE_SIZE))
                success_count += 1

    print(f"Tiles downloaded: {success_count}/{tiles_x * tiles_y}")
    if success_count < (tiles_x * tiles_y) * 0.5:
        print(f"Success rate too low, skipping this region")
        return None
    return big_image


# ==================== 数据变换 ====================

def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """使用 OpenCV 旋转，与训练数据生成一致。"""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        img_array, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(20, 20, 20)
    )

    return Image.fromarray(rotated)


def random_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    if crop_size >= width or crop_size >= height:
        return image.resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    max_x = width - crop_size
    max_y = height - crop_size
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return image.crop((x, y, x + crop_size, y + crop_size))


def add_tile_seams(image: Image.Image) -> Image.Image:
    """瓦片缝隙模拟已禁用 — 真实截图不会有瓦片边界。"""
    return image


def generate_test_samples(base_map: Image.Image, target_class: int, count: int) -> list:
    """从一张大图生成指定数量的测试样本，旋转角度固定为 target_class * 45°。"""
    angle = target_class * 45.0
    samples = []
    for _ in range(count):
        rotated = rotate_image(base_map, angle)
        crop_size = random.randint(CROP_SIZE_MIN, min(CROP_SIZE_MAX, base_map.size[0], base_map.size[1]))
        cropped = random_crop(rotated, crop_size)
        final = cropped.resize((FINAL_SAVE_SIZE, FINAL_SAVE_SIZE), Image.Resampling.BILINEAR)
        samples.append(final)
    return samples


# ==================== 主逻辑 ====================

def create_directory_structure():
    for class_idx in range(TARGET_CLASSES):
        dir_path = OUTPUT_DIR / str(class_idx)
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Directory structure created: {OUTPUT_DIR}")


def generate_real_test_dataset():
    create_directory_structure()

    # 使用与训练集不同的城市，测试泛化能力
    test_cities = [
        (40.7128, -74.0060),   # New York
        (35.6762, 139.6503),   # Tokyo
        (48.8566, 2.3522),     # Paris
        (55.7558, 37.6173),    # Moscow
        (22.3193, 114.1694),   # Hong Kong
        (1.3521, 103.8198),    # Singapore
        (-33.8688, 151.2093),  # Sydney
        (51.5074, -0.1278),    # London
        (37.7749, -122.4194),  # San Francisco
        (25.2048, 55.2708),    # Dubai
    ]

    total = IMAGES_PER_CLASS_TEST * TARGET_CLASSES
    print(f"Plan: {total} test images ({IMAGES_PER_CLASS_TEST} per class x {TARGET_CLASSES} classes)")
    print(f"Test cities (different from training): {[c[2:] for c in [(0,0)]+test_cities]}")

    # 每个城市下载一张大图，从中为所有类别生成样本
    for city_idx, (lat, lon) in enumerate(test_cities):
        print(f"\n=== City {city_idx+1}/{len(test_cities)}: ({lat:.4f}, {lon:.4f}) ===")
        zoom = random.choice([16, 17, 18])
        source_idx = random.choice([0, 0, 0, 1, 1, 2])  # 优先 Esri/Bing

        base_map = download_map_region(lat, lon, zoom, tiles_x=3, tiles_y=3, source_idx=source_idx)
        if base_map is None:
            print(f"Skipping city {city_idx}")
            continue

        # 从每张大图为每个类别生成样本
        samples_per_class = IMAGES_PER_CLASS_TEST // len(test_cities)
        if samples_per_class < 1:
            samples_per_class = 1

        for class_idx in range(TARGET_CLASSES):
            samples = generate_test_samples(base_map, class_idx, samples_per_class)
            for i, img in enumerate(samples):
                # 使用全局计数器避免文件名冲突
                global_idx = city_idx * TARGET_CLASSES * samples_per_class + class_idx * samples_per_class + i
                save_path = OUTPUT_DIR / str(class_idx) / f"real_{global_idx:04d}.png"
                img.save(save_path, "PNG")

        print(f"Generated {samples_per_class} samples per class from this city")
        del base_map

    # 统计
    print("\n" + "=" * 50)
    print("Real Test Dataset Statistics")
    print("=" * 50)
    total_images = 0
    for class_idx in range(TARGET_CLASSES):
        class_dir = OUTPUT_DIR / str(class_idx)
        count = len(list(class_dir.glob("*.png")))
        total_images += count
        print(f"  test/{class_idx} ({class_idx*45}°): {count} images")
    print(f"  Total: {total_images} images")
    print("=" * 50)


if __name__ == "__main__":
    print("=" * 60)
    print("Oasis-Map-Orientation-Surveyor Real Test Data Generator (v3)")
    print("=" * 60)
    random.seed(123)  # 不同于训练的种子
    np.random.seed(123)
    generate_real_test_dataset()
