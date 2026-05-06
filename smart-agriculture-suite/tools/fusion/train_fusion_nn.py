#!/usr/bin/env python3
"""
FusionModule Neural Network — Training Script
===============================================

Trains a small feedforward neural network (5→8→3) for irrigation decision
fusion on ESP32-S3. The trained weights are exported as C arrays that can be
pasted directly into FusionModule.cpp's initNetwork() method.

Architecture
------------
Input(5): normalized sensor values [AirTemp, AirHumi, SoilHumi, Light, Liquid]
  -> Dense(8, ReLU)
  -> Dense(3, Softmax)  [None, Moderate, Heavy]

Training Data Format
--------------------
CSV file with columns:
    air_temp, air_humi, soil_humi, light, liquid, label

Labels: 0=None, 1=Moderate, 2=Heavy

Usage
-----
    python train_fusion_nn.py --data training_data.csv --output weights.h

If no training data is available, the script can generate synthetic data
from rule-based heuristics for bootstrapping:

    python train_fusion_nn.py --synthetic --output weights.h
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print(
        "ERROR: TensorFlow 2.x is required.\n"
        "Install with:  pip install tensorflow"
    )
    sys.exit(1)


# ===================================================================
# Synthetic data generation
# ===================================================================
def generate_synthetic_data(n_samples=5000, seed=42):
    """Generate synthetic irrigation training data from heuristics."""
    rng = np.random.RandomState(seed)

    air_temp = rng.uniform(5, 45, n_samples)
    air_humi = rng.uniform(10, 100, n_samples)
    soil_humi = rng.uniform(5, 100, n_samples)
    light = rng.uniform(0, 10000, n_samples)
    liquid = rng.uniform(0, 100, n_samples)

    labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Need factors (mirrors FusionModule::performFusion logic)
        need_temp = air_temp[i] / 40.0
        need_humi = 1.0 - air_humi[i] / 100.0
        need_soil = 1.0 - soil_humi[i] / 100.0
        need_light = (light[i] / 10000.0) * 0.5
        need_liquid = liquid[i] / 100.0

        # Weighted score (soil gets 30%, temp 20%, humi 20%, light 15%, liquid 15%)
        score = (need_temp * 0.20 + need_humi * 0.20 + need_soil * 0.30 +
                 need_light * 0.15 + need_liquid * 0.15) * 100.0

        # Add some noise for realism
        score += rng.normal(0, 5)

        # Low liquid = no irrigation (safety)
        if liquid[i] < 20:
            labels[i] = 0
        elif score > 65:
            labels[i] = 2  # Heavy
        elif score > 35:
            labels[i] = 1  # Moderate
        else:
            labels[i] = 0  # None

    # Normalize inputs to [0, 1]
    X = np.column_stack([
        air_temp / 40.0,
        air_humi / 100.0,
        soil_humi / 100.0,
        light / 10000.0,
        liquid / 100.0,
    ])

    return X, labels


def load_csv_data(csv_path):
    """Load training data from a CSV file."""
    X_list = []
    y_list = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            X_list.append([
                float(row['air_temp']) / 40.0,
                float(row['air_humi']) / 100.0,
                float(row['soil_humi']) / 100.0,
                float(row['light']) / 10000.0,
                float(row['liquid']) / 100.0,
            ])
            y_list.append(int(row['label']))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)


# ===================================================================
# Model
# ===================================================================
def build_model():
    """Build the 5→8→3 fusion network matching FusionModule architecture."""
    inputs = keras.Input(shape=(5,), name="sensors")
    x = layers.Dense(8, activation="relu", name="hidden")(inputs)
    outputs = layers.Dense(3, activation="softmax", name="output")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="fusion_nn")


# ===================================================================
# Weight export
# ===================================================================
def export_weights_as_c(model, output_path):
    """Export trained weights as C arrays for FusionModule::initNetwork()."""
    layers = model.layers

    # Find hidden and output dense layers
    hidden_layer = None
    output_layer = None
    for layer in layers:
        if isinstance(layer, keras.layers.Dense):
            if hidden_layer is None:
                hidden_layer = layer
            else:
                output_layer = layer

    if hidden_layer is None or output_layer is None:
        print("ERROR: Could not find Dense layers in model")
        sys.exit(1)

    # Extract weights
    weights_ih = hidden_layer.get_weights()[0]  # shape (5, 8)
    bias_h = hidden_layer.get_weights()[1]       # shape (8,)
    weights_ho = output_layer.get_weights()[0]   # shape (8, 3)
    bias_o = output_layer.get_weights()[1]       # shape (3,)

    lines = []
    lines.append("// Auto-generated by tools/fusion/train_fusion_nn.py")
    lines.append("// Paste into FusionModule::initNetwork() to replace preset weights")
    lines.append("")
    lines.append(f"// weightsIH_ (input -> hidden), trained {model.history.history.get('loss', ['N/A'])}")
    lines.append("const float presetIH[kSensorCount][kHidden] = {")
    for i in range(weights_ih.shape[0]):
        row = ", ".join(f"{v:.4f}f" for v in weights_ih[i])
        lines.append(f"    {{{row}}},")
    lines.append("};")
    lines.append("")

    lines.append("// biasH_ (hidden bias)")
    lines.append("const float presetBH[kHidden] = {")
    row = ", ".join(f"{v:.4f}f" for v in bias_h)
    lines.append(f"    {row}")
    lines.append("};")
    lines.append("")

    lines.append("// weightsHO_ (hidden -> output)")
    lines.append("const float presetHO[kHidden][kOutputs] = {")
    for i in range(weights_ho.shape[0]):
        row = ", ".join(f"{v:.4f}f" for v in weights_ho[i])
        lines.append(f"    {{{row}}},")
    lines.append("};")
    lines.append("")

    lines.append("// biasO_ (output bias)")
    lines.append("const float presetBO[kOutputs] = {")
    row = ", ".join(f"{v:.4f}f" for v in bias_o)
    lines.append(f"    {row}")
    lines.append("};")

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Weight header saved to {out}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train FusionModule neural network and export weights as C arrays."
    )
    parser.add_argument("--data", help="CSV training data file")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic training data from heuristics")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="fusion_weights.h",
                        help="Output C header file for weights (default: fusion_weights.h)")
    args = parser.parse_args()

    if not args.data and not args.synthetic:
        print("ERROR: Specify --data <csv> or --synthetic")
        sys.exit(1)

    # Load data
    if args.synthetic:
        print("[STEP 1/4] Generating synthetic training data ...")
        X, y = generate_synthetic_data()
    else:
        print(f"[STEP 1/4] Loading data from {args.data} ...")
        X, y = load_csv_data(args.data)

    print(f"  Samples: {len(y)}, Classes: {np.bincount(y)}")

    # Split
    split = int(len(y) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Build model
    print("[STEP 2/4] Building model ...")
    model = build_model()
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train
    print(f"[STEP 3/4] Training for {args.epochs} epochs ...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        ],
    )

    # Evaluate
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Validation accuracy: {acc:.4f}")

    # Export
    print("[STEP 4/4] Exporting weights ...")
    export_weights_as_c(model, args.output)
    print("[DONE]")


if __name__ == "__main__":
    main()
