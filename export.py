import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from config import IMAGE_SIZE, NUM_CLASSES, ONNX_CONFIG
from model import build_model


def export_to_onnx(model_path: str, output_path: str = "rotation_model.onnx", image_size: int = None):
    """
    将训练好的模型导出为 ONNX 格式
    """
    if image_size is None:
        image_size = IMAGE_SIZE

    device = torch.device("cpu")

    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, requires_grad=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=ONNX_CONFIG["opset_version"],
        do_constant_folding=ONNX_CONFIG["do_constant_folding"],
        input_names=ONNX_CONFIG["input_names"],
        output_names=ONNX_CONFIG["output_names"],
        dynamic_axes=ONNX_CONFIG["dynamic_axes"],
    )

    print(f"Model exported to {output_path}")

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"ONNX model size: {file_size:.2f} MB")

    # 尝试使用 onnxsim 简化模型
    try:
        import onnx
        from onnxsim import simplify

        onnx_model = onnx.load(output_path)
        model_simp, check = simplify(onnx_model)
        if check:
            simp_path = output_path.replace(".onnx", "_sim.onnx")
            onnx.save(model_simp, simp_path)
            simp_size = Path(simp_path).stat().st_size / (1024 * 1024)
            print(f"Simplified model saved to {simp_path} ({simp_size:.2f} MB)")
        else:
            print("Simplification check failed, keeping original model")
    except ImportError:
        print("onnxsim not installed, skipping simplification")


def verify_onnx_model(onnx_path: str, model_path: str = None, image_size: int = None):
    """
    验证导出的 ONNX 模型是否可以正确推理
    """
    if image_size is None:
        image_size = IMAGE_SIZE
    if model_path is None:
        model_path = "best_model.pth"

    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    output = session.run(None, {"input": dummy_input})
    print(f"ONNX model output shape: {output[0].shape}")
    print(f"ONNX model output sample: {output[0][0]}")

    # 验证 PyTorch 和 ONNX 输出一致性
    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    pt_output = model(torch.from_numpy(dummy_input)).detach().numpy()

    max_diff = np.max(np.abs(pt_output - output[0]))
    print(f"PyTorch vs ONNX max output difference: {max_diff:.8f}")
    if max_diff < 1e-5:
        print("PyTorch and ONNX outputs match!")
    else:
        print("WARNING: PyTorch and ONNX outputs differ significantly!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", type=str, default="best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="rotation_model.onnx",
                        help="Output ONNX path")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                        help="Input image size")
    args = parser.parse_args()

    model_path = Path(args.model)
    if model_path.exists():
        export_to_onnx(str(model_path), args.output, args.image_size)

        try:
            verify_onnx_model(args.output, str(model_path), args.image_size)
        except ImportError:
            print("onnxruntime not installed, skipping verification")
    else:
        print(f"Model file not found: {model_path}")
        print("Please run train.py first to train and save the model.")
