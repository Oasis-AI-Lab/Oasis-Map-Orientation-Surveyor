"""
synthetic"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
"""

import os
import math
import random
import time"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
"""

import os
import math
import random
import time
import io
from pathlib import Path
from"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
"""

import os
import math
import random
import time
import io
from pathlib import Path
from typing import Tuple, Optional

import numpy as"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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
"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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


"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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
"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreet"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  #"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°,"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN ="""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25%"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  #"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80),"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))  # 灰色到深灰色
SEAM_WIDTH"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))  # 灰色到深灰色
SEAM_WIDTH_MIN = 1
SEAM_WIDTH_MAX ="""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))  # 灰色到深灰色
SEAM_WIDTH_MIN = 1
SEAM_WIDTH_MAX = 3

# 输出目录
OUTPUT"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))  # 灰色到深灰色
SEAM_WIDTH_MIN = 1
SEAM_WIDTH_MAX = 3

# 输出目录
OUTPUT_DIR = Path("e:/github projects/Oasis"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))  # 灰色到深灰色
SEAM_WIDTH_MIN = 1
SEAM_WIDTH_MAX = 3

# 输出目录
OUTPUT_DIR = Path("e:/github projects/OasisCompany/Oasis-Map-Orientation-Surveyor/dataset")

# 请求配置
"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))  # 灰色到深灰色
SEAM_WIDTH_MIN = 1
SEAM_WIDTH_MAX = 3

# 输出目录
OUTPUT_DIR = Path("e:/github projects/OasisCompany/Oasis-Map-Orientation-Surveyor/dataset")

# 请求配置
REQUEST_TIMEOUT = 30
REQUEST_DELAY ="""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))  # 灰色到深灰色
SEAM_WIDTH_MIN = 1
SEAM_WIDTH_MAX = 3

# 输出目录
OUTPUT_DIR = Path("e:/github projects/OasisCompany/Oasis-Map-Orientation-Surveyor/dataset")

# 请求配置
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.1  # 请求间隔，避免"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))  # 灰色到深灰色
SEAM_WIDTH_MIN = 1
SEAM_WIDTH_MAX = 3

# 输出目录
OUTPUT_DIR = Path("e:/github projects/OasisCompany/Oasis-Map-Orientation-Surveyor/dataset")

# 请求配置
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.1  # 请求间隔，避免被封
HEADERS = {
    "User"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))  # 灰色到深灰色
SEAM_WIDTH_MIN = 1
SEAM_WIDTH_MAX = 3

# 输出目录
OUTPUT_DIR = Path("e:/github projects/OasisCompany/Oasis-Map-Orientation-Surveyor/dataset")

# 请求配置
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.1  # 请求间隔，避免被封
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x"""
synthetic_generator.py
用于生成 Oasis-Map-Orientation-Surveyor 模型的训练数据。
从 OpenStreetMap (OSM) 下载地图切片，合成大图后，
进行随机旋转、随机裁剪，并模拟瓦片交界缝隙，
最终按相对旋转方向（0-7）分类存储到 train/val 目录。
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

# 地图切片服务配置 (OpenStreetMap 标准切片)
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256  # 标准瓦片大小
MAX_ZOOM = 18
MIN_ZOOM = 15

# 数据生成配置
TARGET_CLASSES = 8  # 0-7 对应 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
IMAGES_PER_CLASS_TRAIN = 300
IMAGES_PER_CLASS_VAL = 100
TRAIN_RATIO = 0.75  # 75% 训练集，25% 验证集

# 图像尺寸配置
BASE_IMAGE_SIZE = 1024  # 合成的大图尺寸
CROP_SIZE_MIN = 512     # 随机裁剪最小尺寸
CROP_SIZE_MAX = 1024    # 随机裁剪最大尺寸
FINAL_SAVE_SIZE = 512   # 最终保存尺寸

# 瓦片缝隙模拟配置
SEAM_PROBABILITY = 0.7  # 70% 概率添加缝隙
SEAM_COUNT_MIN = 1
SEAM_COUNT_MAX = 2
SEAM_COLOR_RANGE = ((80, 80, 80), (40, 40, 40))  # 灰色到深灰色
SEAM_WIDTH_MIN = 1
SEAM_WIDTH_MAX = 3

# 输出目录
OUTPUT_DIR = Path("e:/github projects/OasisCompany/Oasis-Map-Orientation-Surveyor/dataset")

# 请求配置
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.1  # 请求间隔，避免被封
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (