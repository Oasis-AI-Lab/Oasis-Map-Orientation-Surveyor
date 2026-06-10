"""
Oasis Map Orientation Surveyor - FastAPI 推理服务
生产级 REST API，支持单图推理和批量推理
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import io
import base64
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import RotationPredictor
from config import ONNX_PATH, CLASS_NAMES

# 全局模型实例
predictor: Optional[RotationPredictor] = None


class PredictionResult(BaseModel):
    class_idx: int
    angle: int
    class_name: str
    confidence: float
    probabilities: dict


class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]
    count: int


class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global predictor
    # 启动时加载模型
    if ONNX_PATH.exists():
        predictor = RotationPredictor(str(ONNX_PATH))
        print(f"Model loaded: {ONNX_PATH}")
    else:
        print(f"Warning: Model not found at {ONNX_PATH}")
    yield
    # 关闭时清理
    predictor = None


app = FastAPI(
    title="Oasis Map Orientation Surveyor API",
    description="地图相对旋转方向检测 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _predict_image(image: Image.Image) -> PredictionResult:
    """内部预测函数"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = predictor.predict(image)
    return PredictionResult(
        class_idx=result["class"],
        angle=result["angle"],
        class_name=result["class_name"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """健康检查端点"""
    return HealthCheck(
        status="healthy",
        model_loaded=predictor is not None,
        model_path=str(ONNX_PATH),
    )


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """
    单图推理端点
    
    - **file**: 地图截图图片（PNG/JPG）
    - 返回预测的旋转方向
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return _predict_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    批量推理端点
    
    - **files**: 多张地图截图图片
    - 返回每张图片的预测结果
    """
    results = []
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            results.append(_predict_image(image))
        except Exception as e:
            results.append(PredictionResult(
                class_idx=-1,
                angle=-1,
                class_name="error",
                confidence=0.0,
                probabilities={},
            ))
    
    return BatchPredictionResult(results=results, count=len(results))


@app.post("/predict/base64", response_model=PredictionResult)
async def predict_base64(image_base64: str):
    """
    Base64 图片推理端点
    
    - **image_base64**: Base64 编码的图片字符串
    - 返回预测的旋转方向
    """
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return _predict_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
