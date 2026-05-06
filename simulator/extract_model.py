#!/usr/bin/env python3
"""
Extract TFLite model bytes from the C header file plant_disease_model.h.
Output: data/plant_disease_model.tflite
"""

import re
import os

HEADER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "smart-agriculture-suite", "include", "plant_disease_model.h",
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "plant_disease_model.tflite")


def extract():
    header = os.path.abspath(HEADER_PATH)
    if not os.path.exists(header):
        print(f"Header not found: {header}")
        return False

    with open(header, "r") as f:
        content = f.read()

    # Find all hex bytes in the array
    pattern = r"0x([0-9a-fA-F]{2})"
    matches = re.findall(pattern, content)
    if not matches:
        print("No hex bytes found in header")
        return False

    data = bytes(int(m, 16) for m in matches)
    print(f"Extracted {len(data)} bytes from header")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        f.write(data)
    print(f"Written to {os.path.abspath(OUTPUT_PATH)}")
    return True


if __name__ == "__main__":
    extract()
