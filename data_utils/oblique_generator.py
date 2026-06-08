"""
oblique_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的斜视卫星图训练数据。
从多个公开地图服务下载斜视/倾斜摄影卫星图，进行随机旋转、裁剪，
模拟瓦片交界缝隙，最终按相对旋转方向（0-7）分类存储。

斜视卫星图特点：
- 从45度角拍摄，能看到建筑物侧面
- 有明确的朝向（北、东北、东、东南等8个方向）
- 用于3D重建时提供立面信息
"""

import os
import math
import random
import time
import io
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import requests
from tqdm import tqdm


# ==================== 配置参数 ====================

# 多地图源配置（支持斜视/倾斜摄影）
TILE_SOURCES = [
    {
        "name": "Google Satellite",
        "url": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "type": "satellite",
    },
    {
        "name": "Google Hybrid",
        "url": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "type": "hybrid",
    },
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
        "name": "CartoDB Dark",
        "url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "subdomains": ["a", "b", "c", "d"],
        "type": "map",
    },
]

TILE_SIZE = 256
MAX_ZOOM = 19
MIN_ZOOM = 16  # 斜视图需要更高zoom

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100

# 图像尺寸配置
CROP_SIZE_MIN = 512
CROP_SIZE_MAX = 1024
FINAL_SAVE_SIZE = 512

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))
SEAM_WIDTH_MIN = 1
SEAM_WIDTH_MAX = 3

# 输出目录
OUTPUT_DIR = Path("e:/github projects/OasisCompany/Oasis-Map-Orientation-Surveyor/dataset")

# 请求配置
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.5
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

MAX_RETRIES = 3


# ==================== 坐标转换工具 ====================

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """将经纬度转换为瓦片坐标 (x, y)。"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """将瓦片坐标转换为左上角的经纬度。"""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def tile_xy_to_quadkey(x: int, y: int, z: int) -> str:
    """将瓦片坐标转换为 Bing Maps 的 quadkey。"""
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
    """根据地图源索引生成瓦片 URL。"""
    source = TILE_SOURCES[source_idx % len(TILE_SOURCES)]
    url = source["url"]
    
    if source.get("use_quadkey"):
        quadkey = tile_xy_to_quadkey(x, y, z)
        url = url.replace("{quadkey}", quadkey)
    else:
        if source.get("subdomains"):
            subdomain = random.choice(source["subdomains"])
            url = url.replace("{s}", subdomain)
        else:
            url = url.replace("{s}.", "")
        url = url.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
    
    return url


def download_tile(x: int, y: int, z: int, source_idx: int = 0, retries: int = MAX_RETRIES) -> Optional[Image.Image]:
    """下载单个地图瓦片，支持多源切换。"""
    for attempt in range(retries):
        try:
            current_source_idx = (source_idx + attempt) % len(TILE_SOURCES)
            url = get_tile_url(current_source_idx, x, y, z)
            
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                print(f"非图片响应 ({x}, {y}, {z}): {content_type}")
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
                print(f"下载瓦片失败 ({x}, {y}, {z}): {e}")
                return None
    return None


def download_map_region(
    center_lat: float,
    center_lon: float,
    zoom: int,
    tiles_x: int = 5,
    tiles_y: int = 5,
    source_idx: int = 0
) -> Optional[Image.Image]:
    """下载指定区域的地图瓦片并拼接成一张大图。"""
    center_x, center_y = deg2num(center_lat, center_lon, zoom)
    
    start_x = center_x - tiles_x // 2
    start_y = center_y - tiles_y // 2
    
    total_width = tiles_x * TILE_SIZE
    total_height = tiles_y * TILE_SIZE
    
    big_image = Image.new("RGB", (total_width, total_height), color=(30, 30, 30))
    
    source_name = TILE_SOURCES[source_idx % len(TILE_SOURCES)]["name"]
    print(f"正在下载卫星图区域: 中心 ({center_lat:.4f}, {center_lon:.4f}), Zoom={zoom}, 源={source_name}")
    
    success_count = 0
    for dy in range(tiles_y):
        for dx in range(tiles_x):
            tile_x = start_x + dx
            tile_y = start_y + dy
            
            tile = download_tile(tile_x, tile_y, zoom, source_idx=source_idx)
            if tile is not None:
                paste_x = dx * TILE_SIZE
                paste_y = dy * TILE_SIZE
                big_image.paste(tile, (paste_x, paste_y))
                success_count += 1
    
    print(f"瓦片下载成功: {success_count}/{tiles_x * tiles_y}")
    
    if success_count < (tiles_x * tiles_y) * 0.5:
        print(f"瓦片下载成功率过低，放弃此区域")
        return None
    
    return big_image


# ==================== 斜视效果模拟 ====================

def simulate_oblique_view(image: Image.Image, direction: int) -> Image.Image:
    """
    模拟斜视效果（倾斜摄影）。
    direction: 0=北, 1=东北, 2=东, 3=东南, 4=南, 5=西南, 6=西, 7=西北
    通过透视变换模拟从该方向45度角观察的效果。
    """
    width, height = image.size
    
    # 定义透视变换矩阵，模拟从不同方向斜视
    # 这里使用简化的透视变换，实际倾斜摄影需要更复杂的3D变换
    
    # 根据方向确定倾斜轴
    if direction in [0, 4]:  # 北/南 - 沿Y轴倾斜
        # 模拟从北或南方向看
        tilt_factor = 0.3 if direction == 0 else -0.3
        coeffs = (
            1.0, 0.0, 0.0,
            tilt_factor, 0.7, 0.0,
            0.0, 0.0, 1.0
        )
    elif direction in [2, 6]:  # 东/西 - 沿X轴倾斜
        tilt_factor = -0.3 if direction == 2 else 0.3
        coeffs = (
            0.7, tilt_factor, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
    elif direction in [1, 5]:  # 东北/西南 - 对角线倾斜
        tilt_x = -0.2 if direction == 1 else 0.2
        tilt_y = 0.2 if direction == 1 else -0.2
        coeffs = (
            0.8, tilt_x, 0.0,
            tilt_y, 0.8, 0.0,
            0.0, 0.0, 1.0
        )
    else:  # 东南/西北
        tilt_x = 0.2 if direction == 3 else -0.2
        tilt_y = 0.2 if direction == 3 else -0.2
        coeffs = (
            0.8, tilt_x, 0.0,
            tilt_y, 0.8, 0.0,
            0.0, 0.0, 1.0
        )
    
    # 应用透视变换
    oblique = image.transform(
        (width, height),
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BILINEAR,
        fillcolor=(20, 20, 20)
    )
    
    return oblique


# ==================== 数据增强与合成 ====================

def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """对图像进行指定角度的旋转。"""
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(20, 20, 20))


def random_crop(image: Image.Image, crop_size: int) -> Image.Image:
    """从图像中随机裁剪指定大小的区域。"""
    width, height = image.size
    if crop_size >= width or crop_size >= height:
        return image.resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    
    max_x = width - crop_size
    max_y = height - crop_size
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    
    return image.crop((x, y, x + crop_size, y + crop_size))


def add_tile_seams(image: Image.Image) -> Image.Image:
    """在图像上随机添加 1-2 条灰色/黑色细线，模拟瓦片交界缝隙。"""
    if random.random() > SEAM_PROBABILITY:
        return image
    
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    num_seams = random.randint(SEAM_COUNT_MIN, SEAM_COUNT_MAX)
    
    for _ in range(num_seams):
        color = (
            random.randint(SEAM_COLOR_RANGE[1][0], SEAM_COLOR_RANGE[0][0]),
            random.randint(SEAM_COLOR_RANGE[1][1], SEAM_COLOR_RANGE[0][1]),
            random.randint(SEAM_COLOR_RANGE[1][2], SEAM_COLOR_RANGE[0][2]),
        )
        line_width = random.randint(SEAM_WIDTH_MIN, SEAM_WIDTH_MAX)
        
        if random.random() > 0.5:
            y = random.randint(0, height)
            draw.line([(0, y), (width, y)], fill=color, width=line_width)
        else:
            x = random.randint(0, width)
            draw.line([(x, 0), (x, height)], fill=color, width=line_width)
    
    return img


def apply_random_transformations(image: Image.Image, target_class: int) -> Image.Image:
    """
    应用随机变换：斜视模拟、旋转、裁剪、添加缝隙。
    target_class: 0-7，对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
    """
    # 1. 模拟斜视效果（方向由 target_class 决定）
    oblique = simulate_oblique_view(image, target_class)
    
    # 2. 旋转（模拟用户截图时的随机旋转）
    angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
    rotated = rotate_image(oblique, angle)
    
    # 3. 随机裁剪
    crop_size = random.randint(CROP_SIZE_MIN, min(CROP_SIZE_MAX, image.size[0], image.size[1]))
    cropped = random_crop(rotated, crop_size)
    
    # 4. 添加瓦片缝隙
    with_seams = add_tile_seams(cropped)
    
    # 5. 调整最终尺寸
    final = with_seams.resize((FINAL_SAVE_SIZE, FINAL_SAVE_SIZE), Image.Resampling.BILINEAR)
    
    return final


# ==================== 数据集生成主逻辑 ====================

def create_directory_structure():
    """创建数据集目录结构。"""
    splits = ["train", "val"]
    for split in splits:
        for class_idx in range(TARGET_CLASSES):
            dir_path = OUTPUT_DIR / split / str(class_idx)
            dir_path.mkdir(parents=True, exist_ok=True)
    print(f"目录结构已创建: {OUTPUT_DIR}")


def generate_synthetic_dataset():
    """主函数：生成完整的斜视卫星图合成数据集。"""
    create_directory_structure()
    
    # 城市中心坐标（选择有高楼建筑的城市，斜视效果更明显）
    city_centers = [
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
    ]
    
    total_train = IMAGES_PER_CLASS_TRAIN * TARGET_CLASSES
    total_val = IMAGES_PER_CLASS_VAL * TARGET_CLASSES
    
    print(f"计划生成数据: {total_train} 张训练 + {total_val} 张验证 = {total_train + total_val} 张")
    print(f"使用地图源: {[s['name'] for s in TILE_SOURCES]}")
    
    for class_idx in range(TARGET_CLASSES):
        print(f"\n正在生成分类 {class_idx} (方向 {class_idx * 45}°) 的数据...")
        
        city_idx = class_idx % len(city_centers)
        center_lat, center_lon = city_centers[city_idx]
        zoom = random.randint(MIN_ZOOM, MAX_ZOOM)
        
        source_idx = class_idx % len(TILE_SOURCES)
        base_map = download_map_region(center_lat, center_lon, zoom, tiles_x=5, tiles_y=5, source_idx=source_idx)
        
        if base_map is None:
            print(f"无法下载地图区域，跳过类别 {class_idx}")
            continue
        
        # 生成训练集
        for i in tqdm(range(IMAGES_PER_CLASS_TRAIN), desc=f"类别 {class_idx} 训练集"):
            transformed = apply_random_transformations(base_map, class_idx)
            save_path = OUTPUT_DIR / "train" / str(class_idx) / f"{class_idx}_{i:04d}.png"
            transformed.save(save_path, "PNG")
        
        # 生成验证集
        for i in tqdm(range(IMAGES_PER_CLASS_VAL), desc=f"类别 {class_idx} 验证集"):
            transformed = apply_random_transformations(base_map, class_idx)
            save_path = OUTPUT_DIR / "val" / str(class_idx) / f"{class_idx}_{i:04d}.png"
            transformed.save(save_path, "PNG")
        
        del base_map
    
    print(f"\n数据生成完成！保存在: {OUTPUT_DIR}")
    print_summary()


def print_summary():
    """打印数据集统计信息。"""
    print("\n" + "=" * 50)
    print("数据集生成统计")
    print("=" * 50)
    
    for split in ["train", "val"]:
        split_dir = OUTPUT_DIR / split
        if not split_dir.exists():
            continue
        
        total_images = 0
        for class_idx in range(TARGET_CLASSES):
            class_dir = split_dir / str(class_idx)
            if class_dir.exists():
                count = len(list(class_dir.glob("*.png")))
                total_images += count
                print(f"  {split}/{class_idx}: {count} 张")
        
        print(f"  {split} 总计: {total_images} 张")
    
    print("=" * 50)


# ==================== 入口 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Oasis-Map-Orientation-Surveyor 斜视卫星图合成数据生成器")
    print("=" * 60)
    
    random.seed(42)
    np.random.seed(42)
    
    generate_synthetic_dataset()
