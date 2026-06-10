"""
oblique_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的卫星图旋转分类训练数据。
从多个公开卫星图服务下载地图切片，进行显式旋转、随机裁剪和域随机化，
按旋转方向（0-7）分类存储。

v3 优化：
- 去掉瓦片缝隙模拟（真实截图不会有瓦片边界）
- 增强域随机化（模糊、锐化、色彩偏移等）
- 改用 OpenCV 旋转（更接近真实截图工具）
- 标签 = 旋转角度（target_class * 45°），语义一致
- 每个类别使用多个城市、多个地图源、多个 zoom 级别
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
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import requests
from tqdm import tqdm


# ==================== 配置参数 ====================

# 卫星图源（Esri 最稳定放前面，Google 在中国可能超时）
# 新增中国地图源：高德卫星、百度卫星
TILE_SOURCES = [
    {
        "name": "Esri World Imagery",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    },
    {
        "name": "Bing Aerial",
        "url": "https://t0.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=1",
        "use_quadkey": True,
    },
    {
        "name": "Google Satellite",
        "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    },
    {
        "name": "Google Satellite (mt2)",
        "url": "https://mt2.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    },
    {
        "name": "Google Satellite (mt3)",
        "url": "https://mt3.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    },
    # 中国地图源 - 高德卫星图
    {
        "name": "Amap Satellite",
        "url": "https://webst01.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}",
    },
    {
        "name": "Amap Satellite (02)",
        "url": "https://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}",
    },
    {
        "name": "Amap Satellite (03)",
        "url": "https://webst03.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}",
    },
    # 中国地图源 - 百度卫星图（需坐标转换，使用简化版）
    {
        "name": "Baidu Satellite",
        "url": "https://maponline0.bdimg.com/tile/?qt=vtile&x={x}&y={y}&z={z}&styles=pl&scaler=1&udt=20240101",
    },
]

TILE_SIZE = 256
MAX_ZOOM = 19
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8
IMAGES_PER_CLASS_TRAIN = 1000
IMAGES_PER_CLASS_VAL = 200
CITIES_PER_CLASS = 8
IMAGES_PER_CITY_TRAIN = IMAGES_PER_CLASS_TRAIN // CITIES_PER_CLASS  # 125
IMAGES_PER_CITY_VAL = IMAGES_PER_CLASS_VAL // CITIES_PER_CLASS      # 25
ZOOM_LEVELS = [15, 16, 17, 18, 19]

# 图像尺寸配置
CROP_SIZE_MIN = 512
CROP_SIZE_MAX = 1024
FINAL_SAVE_SIZE = 512

# 输出目录
OUTPUT_DIR = Path("e:/github projects/OasisCompany/Oasis-Map-Orientation-Surveyor/dataset")

# 请求配置
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 0.1
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
MAX_RETRIES = 3

# 40+ 全球城市，覆盖不同地形和建筑风格
GLOBAL_CITIES = [
    # 中国
    (39.9042, 116.4074),   # 北京
    (31.2304, 121.4737),   # 上海
    (30.5728, 104.0668),   # 成都
    (23.1291, 113.2644),   # 广州
    (29.5630, 106.5516),   # 重庆
    (34.3416, 108.9398),   # 西安
    (36.0671, 120.3826),   # 青岛
    (26.0745, 119.2965),   # 福州
    (45.8038, 126.5350),   # 哈尔滨
    (43.8256, 87.6168),    # 乌鲁木齐
    (22.5431, 114.0579),   # 深圳
    (30.2741, 120.1551),   # 杭州
    # 亚洲
    (35.6762, 139.6503),   # Tokyo
    (37.5665, 126.9780),   # Seoul
    (22.3193, 114.1694),   # Hong Kong
    (1.3521, 103.8198),    # Singapore
    (13.7563, 100.5018),   # Bangkok
    (19.0760, 72.8777),    # Mumbai
    (28.6139, 77.2090),    # New Delhi
    # 欧洲
    (48.8566, 2.3522),     # Paris
    (51.5074, -0.1278),    # London
    (52.5200, 13.4050),    # Berlin
    (41.9028, 12.4964),    # Rome
    (40.4168, -3.7038),    # Madrid
    (59.3293, 18.0686),    # Stockholm
    (55.7558, 37.6173),    # Moscow
    (50.0755, 14.4378),    # Prague
    # 北美
    (40.7128, -74.0060),   # New York
    (34.0522, -118.2437),  # Los Angeles
    (37.7749, -122.4194),  # San Francisco
    (41.8781, -87.6298),   # Chicago
    (25.7617, -80.1918),   # Miami
    (43.6532, -79.3832),   # Toronto
    # 南美/大洋洲/中东/非洲
    (-33.8688, 151.2093),  # Sydney
    (-23.5505, -46.6333),  # Sao Paulo
    (25.2048, 55.2708),    # Dubai
    (30.0444, 31.2357),    # Cairo
    (-1.2921, 36.8219),    # Nairobi
    (6.5244, 3.3792),      # Lagos
]


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
    center_lat: float, center_lon: float, zoom: int,
    tiles_x: int = 3, tiles_y: int = 3, source_idx: int = 0
) -> Optional[Image.Image]:
    center_x, center_y = deg2num(center_lat, center_lon, zoom)
    start_x = center_x - tiles_x // 2
    start_y = center_y - tiles_y // 2
    total_width = tiles_x * TILE_SIZE
    total_height = tiles_y * TILE_SIZE
    big_image = Image.new("RGB", (total_width, total_height), color=(30, 30, 30))

    source_name = TILE_SOURCES[source_idx % len(TILE_SOURCES)]["name"]
    print(f"  Downloading ({center_lat:.4f}, {center_lon:.4f}) Z{zoom} [{source_name}]")

    success_count = 0
    for dy in range(tiles_y):
        for dx in range(tiles_x):
            tile = download_tile(start_x + dx, start_y + dy, zoom, source_idx=source_idx)
            if tile is not None:
                big_image.paste(tile, (dx * TILE_SIZE, dy * TILE_SIZE))
                success_count += 1

    if success_count < (tiles_x * tiles_y) * 0.5:
        print(f"  Success rate too low ({success_count}/{tiles_x*tiles_y}), skipping")
        return None

    return big_image


# ==================== 数据变换 ====================

def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """
    使用 OpenCV 进行旋转，更接近真实截图的变换方式。
    相比 PIL rotate，OpenCV 的 warpAffine 插值实现不同，
    可减少合成数据与真实数据之间的插值差异。
    """
    # PIL -> numpy (RGB)
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后图像尺寸（保持完整内容）
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵中心点
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 执行旋转
    rotated = cv2.warpAffine(
        img_array, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(20, 20, 20)
    )

    # numpy -> PIL
    return Image.fromarray(rotated)


def random_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    if crop_size >= width or crop_size >= height:
        return image.resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    x = random.randint(0, width - crop_size)
    y = random.randint(0, height - crop_size)
    return image.crop((x, y, x + crop_size, y + crop_size))


def add_tile_seams(image: Image.Image) -> Image.Image:
    """瓦片缝隙模拟已禁用 — 真实截图不会有瓦片边界。"""
    return image


def apply_domain_randomization(image: Image.Image) -> Image.Image:
    """增强域随机化：模拟更丰富的真实采集条件变化。"""
    img = image.copy()

    # 1. 随机亮度/对比度/饱和度
    if random.random() > 0.2:
        brightness = random.uniform(0.6, 1.4)
        contrast = random.uniform(0.6, 1.4)
        saturation = random.uniform(0.6, 1.4)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)

    # 2. 随机高斯模糊 (模拟低分辨率/失焦)
    if random.random() > 0.5:
        radius = random.uniform(0.3, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 3. 随机锐化 (模拟过度锐化的图像)
    if random.random() > 0.7:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # 4. 随机色彩偏移 (模拟不同传感器白平衡)
    if random.random() > 0.6:
        img_array = np.array(img, dtype=np.float32)
        for c in range(3):
            shift = random.uniform(-15, 15)
            img_array[:, :, c] = np.clip(img_array[:, :, c] + shift, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))

    # 5. 随机高斯噪声
    if random.random() > 0.4:
        img_array = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(2, 15), img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))

    # 6. 随机 JPEG 压缩
    if random.random() > 0.3:
        quality = random.randint(50, 95)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")

    # 7. 随机色调偏移 (模拟不同时间拍摄)
    if random.random() > 0.8:
        img_array = np.array(img, dtype=np.float32)
        img_array = np.clip(img_array * random.uniform(0.9, 1.1), 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))

    return img


def apply_random_transformations(image: Image.Image, target_class: int) -> Image.Image:
    """
    纯旋转任务：标签 = target_class * 45°。
    不做斜视模拟，不做随机旋转，保证标签语义一致。
    """
    angle = target_class * 45.0
    rotated = rotate_image(image, angle)

    # 从旋转后图像中心区域裁剪，避免黑色边界
    crop_size = random.randint(CROP_SIZE_MIN, min(CROP_SIZE_MAX, rotated.size[0], rotated.size[1]))
    width, height = rotated.size

    # 计算安全裁剪区域（避开可能的黑色边界）
    orig_w, orig_h = image.size
    margin_x = max(0, (width - orig_w) // 2)
    margin_y = max(0, (height - orig_h) // 2)

    min_x = margin_x
    max_x = width - crop_size - margin_x
    min_y = margin_y
    max_y = height - crop_size - margin_y

    if max_x > min_x and max_y > min_y:
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
    else:
        x = random.randint(0, max(0, width - crop_size))
        y = random.randint(0, max(0, height - crop_size))

    cropped = rotated.crop((x, y, x + crop_size, y + crop_size))
    randomized = apply_domain_randomization(cropped)
    final = randomized.resize((FINAL_SAVE_SIZE, FINAL_SAVE_SIZE), Image.Resampling.BILINEAR)
    return final


# ==================== 数据集生成主逻辑 ====================

def create_directory_structure():
    for split in ["train", "val"]:
        for class_idx in range(TARGET_CLASSES):
            (OUTPUT_DIR / split / str(class_idx)).mkdir(parents=True, exist_ok=True)
    print(f"Directory structure created: {OUTPUT_DIR}")


def generate_synthetic_dataset():
    create_directory_structure()

    total_train = IMAGES_PER_CLASS_TRAIN * TARGET_CLASSES
    total_val = IMAGES_PER_CLASS_VAL * TARGET_CLASSES
    print(f"Target: {total_train} train + {total_val} val = {total_train + total_val} images")
    print(f"Cities per class: {CITIES_PER_CLASS}, Total cities available: {len(GLOBAL_CITIES)}")
    print(f"Zoom levels: {ZOOM_LEVELS}")
    print(f"Sources: {[s['name'] for s in TILE_SOURCES]}")

    global_counter = [0]  # 用列表包装以在嵌套函数中修改

    for class_idx in range(TARGET_CLASSES):
        print(f"\n{'='*50}")
        print(f"Class {class_idx} ({class_idx * 45}°)")
        print(f"{'='*50}")

        selected_cities = random.sample(GLOBAL_CITIES, min(CITIES_PER_CLASS, len(GLOBAL_CITIES)))

        for city_idx, (lat, lon) in enumerate(selected_cities):
            zoom = random.choice(ZOOM_LEVELS)
            source_idx = random.choice([0, 0, 0, 1, 1, 2])  # 优先 Esri/Bing

            base_map = download_map_region(lat, lon, zoom, source_idx=source_idx)
            if base_map is None:
                continue

            # 训练集
            for i in tqdm(range(IMAGES_PER_CITY_TRAIN), desc=f"  C{class_idx} train [{city_idx+1}/{len(selected_cities)}]", leave=False):
                transformed = apply_random_transformations(base_map, class_idx)
                save_path = OUTPUT_DIR / "train" / str(class_idx) / f"{class_idx}_{global_counter[0]:05d}.png"
                transformed.save(save_path, "PNG")
                global_counter[0] += 1

            # 验证集
            for i in tqdm(range(IMAGES_PER_CITY_VAL), desc=f"  C{class_idx} val [{city_idx+1}/{len(selected_cities)}]", leave=False):
                transformed = apply_random_transformations(base_map, class_idx)
                save_path = OUTPUT_DIR / "val" / str(class_idx) / f"{class_idx}_{global_counter[0]:05d}.png"
                transformed.save(save_path, "PNG")
                global_counter[0] += 1

            del base_map

    print(f"\nData generation complete: {OUTPUT_DIR}")
    print_summary()


def print_summary():
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    for split in ["train", "val"]:
        total = 0
        for class_idx in range(TARGET_CLASSES):
            count = len(list((OUTPUT_DIR / split / str(class_idx)).glob("*.png")))
            total += count
            print(f"  {split}/{class_idx} ({class_idx*45}°): {count}")
        print(f"  {split} total: {total}")
    print("=" * 50)


if __name__ == "__main__":
    print("=" * 60)
    print("Oasis-Map-Orientation-Surveyor Data Generator (v3)")
    print("Domain gap reduction: no seams, enhanced randomization, OpenCV rotate")
    print("=" * 60)
    random.seed(42)
    np.random.seed(42)
    generate_synthetic_dataset()
