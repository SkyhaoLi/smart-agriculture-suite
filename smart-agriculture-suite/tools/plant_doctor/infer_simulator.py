#!/usr/bin/env python3
"""
Plant Disease Detection — Offline Inference Simulator
=====================================================

Simulates the ESP32 PlantDoctorModule inference pipeline on a PC.
Reads images from a directory, applies the same preprocessing as the
firmware (resize 96x96, normalize [-1,1], INT8 quantize), runs the
TFLite model, and prints detection results.

Usage
-----
    python infer_simulator.py --model model.tflite --input ./test_images/
    python infer_simulator.py --model model.tflite --input single_leaf.jpg
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import tflite_runtime.interpreter as tflite
    HAS_TFLITE_RUNTIME = True
except ImportError:
    HAS_TFLITE_RUNTIME = False


# Same labels as PlantDoctorModule.cpp
CLASS_LABELS_EN = [
    "Healthy",
    "Strawberry_Anthracnose",
    "Strawberry_Gray_Mold",
    "Strawberry_Leaf_Scorch",
    "Strawberry_Powdery_Mildew",
]

CLASS_LABELS_CN = [
    "健康",
    "草莓炭疽病",
    "草莓灰霉病",
    "草莓叶焦病",
    "草莓白粉病",
]

TREATMENTS = [
    "叶片健康。保持通风和定期检查。",
    "清除病株残体，避免伤口感染。施用咪鲜胺等杀菌剂。",
    "降低湿度，增加通风。及时摘除病果，施用嘧霉胺等杀菌剂。",
    "移除感染叶片，降低叶面湿度。必要时施用杀菌剂。",
    "移除病叶，改善通风。喷施硫制剂或三唑类杀菌剂。",
]


def load_interpreter(model_path):
    """Load TFLite model, preferring tflite_runtime for smaller footprint."""
    if HAS_TFLITE_RUNTIME:
        interpreter = tflite.Interpreter(model_path=model_path)
    elif HAS_TF:
        interpreter = tf.lite.Interpreter(model_path=model_path)
    else:
        print("Error: need tensorflow or tflite-runtime installed")
        sys.exit(1)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image_path, input_details):
    """
    Replicate PlantDoctorModule::captureAndPreprocess:
      1. Resize to 96x96
      2. Normalize: (pixel / 127.5) - 1.0  →  [-1, 1]
      3. Quantize to int8 using the model's input scale + zero_point
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((96, 96), Image.BILINEAR)
    pixels = np.array(img, dtype=np.float32)

    # Normalize to [-1, 1] — same as firmware
    normalized = (pixels / 127.5) - 1.0

    # Quantize to int8 — same as firmware
    input_scale = input_details["quantization_parameters"]["scales"][0]
    input_zp = input_details["quantization_parameters"]["zero_points"][0]

    quantized = np.round(normalized / input_scale) + input_zp
    quantized = np.clip(quantized, -128, 127).astype(np.int8)

    return quantized


def run_inference(interpreter, input_data, output_details):
    """
    Replicate PlantDoctorModule::runInference:
      1. Invoke interpreter
      2. Dequantize output
      3. Apply softmax manually (same as firmware)
    """
    input_details = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details["index"])[0]

    # Dequantize: (output - zero_point) * scale — same as firmware
    output_scale = output_details["quantization_parameters"]["scales"][0]
    output_zp = output_details["quantization_parameters"]["zero_points"][0]

    confidences = (output_data.astype(np.float32) - output_zp) * output_scale

    # Softmax — same as firmware
    max_val = np.max(confidences)
    exp_vals = np.exp(confidences - max_val)
    softmax_output = exp_vals / np.sum(exp_vals)

    return softmax_output


def collect_image_paths(input_path):
    """Collect image file paths from a file or directory."""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        paths = []
        for f in sorted(os.listdir(input_path)):
            if os.path.splitext(f)[1].lower() in valid_ext:
                paths.append(os.path.join(input_path, f))
        return paths
    else:
        print(f"Error: {input_path} is not a file or directory")
        sys.exit(1)


def format_bar(confidence, width=30):
    """Render a simple ASCII bar chart."""
    filled = int(confidence * width)
    return "[" + "=" * filled + " " * (width - filled) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Plant Disease Detection — Offline Inference Simulator"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to INT8 quantized TFLite model (.tflite)",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Image file or directory of images",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.70,
        help="Confidence threshold for disease alarm (default: 0.70)",
    )
    args = parser.parse_args()

    image_paths = collect_image_paths(args.input)
    if not image_paths:
        print("No images found.")
        return

    print(f"Loading model: {args.model}")
    interpreter = load_interpreter(args.model)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details["shape"]
    w, h, c = int(input_shape[1]), int(input_shape[2]), int(input_shape[3])
    print("Model input:  {}x{}x{}  INT8".format(w, h, c))
    print(f"Model output: {output_details['shape'][1]} classes  INT8")
    print(f"Images:       {len(image_paths)}")
    print(f"Threshold:    {args.threshold:.0%}")
    print("=" * 72)

    total = len(image_paths)
    disease_count = 0

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)

        try:
            input_data = preprocess_image(img_path, input_details)
            input_data = np.expand_dims(input_data, axis=0)  # add batch dim
            confidences = run_inference(interpreter, input_data, output_details)
        except Exception as e:
            print("[}/{}] {} — ERROR: {}".format(idx+1, total, filename, e))
            continue

        disease_id = int(np.argmax(confidences))
        confidence = float(confidences[disease_id])
        is_disease = disease_id > 0 and confidence >= args.threshold

        if is_disease:
            disease_count += 1

        # Header line
        status = "ALERT" if is_disease else "OK"
        print(f"\n[{idx+1}/{total}] {filename}  —  {status}")
        print("-" * 72)
        print(f"  Result: {CLASS_LABELS_EN[disease_id]} ({CLASS_LABELS_CN[disease_id]})")
        print(f"  Confidence: {confidence:.1%}")

        # Per-class breakdown
        print(f"  Class probabilities:")
        for i, (label_en, label_cn) in enumerate(zip(CLASS_LABELS_EN, CLASS_LABELS_CN)):
            marker = " <<<<" if i == disease_id else ""
            print(f"    {i}: {label_en:35s} {label_cn:8s} {confidences[i]:6.1%} {format_bar(confidences[i])}{marker}")

        # Treatment suggestion (same as firmware)
        if is_disease:
            print(f"  Treatment: {TREATMENTS[disease_id]}")

    # Summary
    print("\n" + "=" * 72)
    print(f"Total images: {total}")
    print(f"Disease detected: {disease_count}")
    print(f"Healthy / below threshold: {total - disease_count}")


if __name__ == "__main__":
    main()
