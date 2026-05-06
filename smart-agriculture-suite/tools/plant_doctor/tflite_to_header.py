import argparse
from pathlib import Path


def write_header(model_bytes: bytes, output: Path, symbol: str) -> None:
    lines = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"const unsigned char {symbol}[] = {{")

    row = []
    for index, value in enumerate(model_bytes):
        row.append(f"0x{value:02x}")
        if len(row) == 12 or index == len(model_bytes) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []

    lines.append("};")
    lines.append(f"const unsigned int {symbol}_len = {len(model_bytes)};")
    lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a .tflite model into a C header file.")
    parser.add_argument("--input", required=True, help="Path to the input .tflite file")
    parser.add_argument("--output", required=True, help="Path to the output .h file")
    parser.add_argument("--symbol", default="model_tflite", help="C symbol name")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_bytes = input_path.read_bytes()
    write_header(model_bytes, output_path, args.symbol)

    print(f"Converted {input_path} -> {output_path}")
    print(f"Symbol: {args.symbol}")
    print(f"Bytes: {len(model_bytes)}")


if __name__ == "__main__":
    main()
