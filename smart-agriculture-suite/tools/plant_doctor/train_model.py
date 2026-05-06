#!/usr/bin/env python3
"""
Plant Disease Detection Model — Training Pipeline
===================================================

Trains a lightweight CNN for plant disease classification on strawberry leaves,
then exports an INT8-quantized TFLite model suitable for deployment on ESP32-S3
with TensorFlow Lite for Microcontrollers.

Architecture
------------
Input(96, 96, 3)
  -> Conv2D(16, 3x3, relu) -> MaxPool2D
  -> Conv2D(32, 3x3, relu) -> MaxPool2D
  -> Conv2D(64, 3x3, relu) -> MaxPool2D
  -> Flatten -> Dense(64, relu) -> Dropout(0.3) -> Dense(num_classes, softmax)

Expected data directory layout::

    data_dir/
      Healthy/
      Strawberry_Leaf_Scorch/
      Strawberry_Powdery_Mildew/
      Strawberry_Anthracnose/
      Strawberry_Gray_Mold/

Usage
-----
    python train_model.py --data_dir ./dataset --epochs 30 --batch_size 32 \
                          --output model.tflite --num_classes 5
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# TensorFlow imports — keep these after standard-library imports so that the
# failure message below is shown clearly when TF is not installed.
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print(
        "ERROR: TensorFlow 2.x is required.\n"
        "Install it with:  pip install tensorflow\n"
        "For GPU support:   pip install tensorflow[and-cuda]"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_HEIGHT = 96
IMG_WIDTH = 96
IMG_CHANNELS = 3

# Default class names — used for the training report and label mapping.
DEFAULT_CLASS_NAMES = [
    "Healthy",
    "Strawberry_Anthracnose",
    "Strawberry_Gray_Mold",
    "Strawberry_Leaf_Scorch",
    "Strawberry_Powdery_Mildew",
]


# ===================================================================
# Model definition
# ===================================================================
def build_model(num_classes: int) -> keras.Model:
    """Build the lightweight CNN suitable for 96x96 input on ESP32-S3.

    The architecture is intentionally small so that the INT8-quantized model
    fits comfortably within the ESP32-S3 PSRAM tensor arena (150 KB default).
    """
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name="image")

    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)  # normalize to [-1, 1]

    # Block 1
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Classifier head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="plant_disease_cnn")
    return model


# ===================================================================
# Data loading & augmentation
# ===================================================================
def load_dataset(data_dir: str, batch_size: int, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Load the image dataset from *data_dir* and split into train/validation.

    Returns
    -------
    train_ds : tf.data.Dataset
        Training dataset with augmentation applied.
    val_ds : tf.data.Dataset
        Validation dataset (no augmentation).
    class_names : list[str]
        Ordered list of class names derived from subdirectory names.
    """
    data_path = Path(data_dir).expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Load full dataset — 80/20 split
    train_ds = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"[INFO] Found {len(class_names)} classes: {class_names}")

    # ------------------------------------------------------------------
    # Data augmentation pipeline (applied only to training data)
    # ------------------------------------------------------------------
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", name="rand_flip"),
            layers.RandomRotation(factor=15.0 / 360.0, name="rand_rot"),  # +/-15 deg
            layers.RandomBrightness(factor=0.2, name="rand_brightness"),   # +/-20%
            layers.RandomZoom(height_factor=0.1, name="rand_zoom"),        # +/-10%
        ],
        name="augmentation",
    )

    # Apply augmentation to training split only.
    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Prefetch for performance.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


# ===================================================================
# INT8 quantized TFLite export
# ===================================================================
def export_int8_tflite(model: keras.Model, train_ds, output_path: str) -> dict:
    """Convert the Keras model to an INT8-quantized TFLite flatbuffer.

    A representative dataset (a few batches from the training set) is used
    to calibrate the quantization ranges for all tensors.

    Returns
    -------
    dict with keys ``model_size_bytes`` and ``output_path``.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable INT8 post-training quantization with representative data.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        """Yield a small number of calibrated samples for quantization."""
        count = 0
        for images, _ in train_ds:
            # Ensure float32 input for the calibration step.
            yield [tf.cast(images, tf.float32).numpy()]
            count += 1
            if count >= 100:  # 100 batches is enough for calibration
                break

    converter.representative_dataset = representative_dataset

    # Force full-integer quantization (inputs and outputs also int8).
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_model)

    print(f"[INFO] INT8 TFLite model saved to {out}  ({len(tflite_model)} bytes)")
    return {"model_size_bytes": len(tflite_model), "output_path": str(out)}


# ===================================================================
# Training report
# ===================================================================
def generate_report(
    model: keras.Model,
    history: keras.callbacks.History,
    val_ds,
    class_names: list,
    export_info: dict,
    report_path: str,
) -> None:
    """Write a human-readable training report including metrics and confusion matrix."""
    # Run inference on validation set to build a confusion matrix.
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix.
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    # Per-class accuracy.
    per_class_acc = {}
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        per_class_acc[name] = cm[i, i] / total if total > 0 else 0.0

    # Overall accuracy from last epoch.
    final_epoch = len(history.history["accuracy"])
    best_val_acc = max(history.history["val_accuracy"])
    best_val_loss = min(history.history["val_loss"])

    lines = []
    lines.append("=" * 70)
    lines.append("  PLANT DISEASE DETECTION — TRAINING REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Date              : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"TensorFlow version: {tf.__version__}")
    lines.append(f"Input shape       : ({IMG_HEIGHT}, {IMG_WIDTH}, {IMG_CHANNELS})")
    lines.append(f"Number of classes : {num_classes}")
    lines.append(f"Class names       : {class_names}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  TRAINING SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Epochs trained    : {final_epoch}")
    lines.append(f"Best val accuracy : {best_val_acc:.4f}")
    lines.append(f"Best val loss     : {best_val_loss:.4f}")
    lines.append(f"Final train acc   : {history.history['accuracy'][-1]:.4f}")
    lines.append(f"Final train loss  : {history.history['loss'][-1]:.4f}")
    lines.append(f"Final val acc     : {history.history['val_accuracy'][-1]:.4f}")
    lines.append(f"Final val loss    : {history.history['val_loss'][-1]:.4f}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  PER-CLASS ACCURACY (validation set)")
    lines.append("-" * 70)
    for name in class_names:
        lines.append(f"  {name:<35s}  {per_class_acc[name]:.4f}  ({per_class_acc[name]*100:.1f}%)")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  CONFUSION MATRIX (rows=true, cols=predicted)")
    lines.append("-" * 70)
    # Header row.
    header = "true \\ pred".ljust(35)
    for name in class_names:
        header += f"  {name[:8]:>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, name in enumerate(class_names):
        row = name[:35].ljust(35)
        for j in range(num_classes):
            row += f"  {cm[i, j]:>8}"
        lines.append(row)
    lines.append("")
    lines.append("-" * 70)
    lines.append("  EXPORTED MODEL")
    lines.append("-" * 70)
    lines.append(f"Output path       : {export_info['output_path']}")
    lines.append(f"Model size        : {export_info['model_size_bytes']} bytes "
                 f"({export_info['model_size_bytes'] / 1024:.1f} KB)")
    lines.append(f"Quantization      : INT8 (full-integer)")
    lines.append(f"Input type        : int8")
    lines.append(f"Output type       : int8")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  CLASS LABEL MAPPING (index -> disease name)")
    lines.append("-" * 70)
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx} -> {name}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  FIRMWARE UPDATE INSTRUCTIONS")
    lines.append("-" * 70)
    lines.append("  1. Convert the TFLite model to a C header:")
    lines.append("       python tools/plant_doctor/tflite_to_header.py \\")
    lines.append("         --input model.tflite \\")
    lines.append("         --output include/plant_disease_model.h \\")
    lines.append("         --symbol plant_disease_model_tflite")
    lines.append("")
    lines.append("  2. Update the class labels and treatment strings in")
    lines.append("     src/PlantDoctorModule.cpp to match the new model.")
    lines.append("")
    lines.append("  3. Update PLANT_MODEL_NUM_CLASSES in the generated header")
    lines.append("     to reflect the new class count.")
    lines.append("")
    lines.append("  4. Rebuild and flash the firmware.")
    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    rp = Path(report_path).expanduser().resolve()
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(report_text, encoding="utf-8")
    print(f"[INFO] Training report saved to {rp}")


# ===================================================================
# Callbacks
# ===================================================================
def make_callbacks(output_path: str) -> list:
    """Return a list of Keras callbacks for the training run."""
    base = Path(output_path).expanduser().resolve().parent

    callbacks = [
        # Save the best model (by validation accuracy) as a Keras .keras file.
        keras.callbacks.ModelCheckpoint(
            filepath=str(base / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # Stop early if validation accuracy plateaus.
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate when the loss stagnates.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    return callbacks


# ===================================================================
# Main entry point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train a plant disease detection CNN and export an INT8 TFLite model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root directory of the image dataset. Must contain one subdirectory per class.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32).",
    )
    parser.add_argument(
        "--output",
        default="model.tflite",
        help="Output path for the INT8 quantized TFLite model (default: model.tflite).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Expected number of disease classes (default: 5). Used for validation only; "
             "actual classes are detected from the data directory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--report",
        default="training_report.txt",
        help="Path for the training report text file (default: training_report.txt).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("[STEP 1/5] Loading dataset ...")
    train_ds, val_ds, class_names = load_dataset(args.data_dir, args.batch_size)

    if len(class_names) != args.num_classes:
        print(
            f"[WARNING] Expected {args.num_classes} classes but found "
            f"{len(class_names)}: {class_names}"
        )
        print("          Continuing with the detected classes.")

    num_classes = len(class_names)

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    print("[STEP 2/5] Building model ...")
    model = build_model(num_classes)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print(f"[STEP 3/5] Training for up to {args.epochs} epochs ...")
    callbacks = make_callbacks(args.output)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # 4. Export INT8 TFLite
    # ------------------------------------------------------------------
    print("[STEP 4/5] Exporting INT8 quantized TFLite model ...")
    export_info = export_int8_tflite(model, train_ds, args.output)

    # ------------------------------------------------------------------
    # 5. Generate report
    # ------------------------------------------------------------------
    print("[STEP 5/5] Generating training report ...")
    generate_report(
        model=model,
        history=history,
        val_ds=val_ds,
        class_names=class_names,
        export_info=export_info,
        report_path=args.report,
    )

    print("\n[DONE] Training pipeline complete.")
    print(f"  Model  : {export_info['output_path']}")
    print(f"  Report : {Path(args.report).expanduser().resolve()}")
    print(f"  Size   : {export_info['model_size_bytes']} bytes "
          f"({export_info['model_size_bytes'] / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
