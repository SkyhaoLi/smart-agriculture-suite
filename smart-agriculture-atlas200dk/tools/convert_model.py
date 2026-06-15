#!/usr/bin/env python3
"""
模型转换工具 - TFLite -> ONNX -> OM

用法:
  # TFLite 转 ONNX (在PC上运行)
  python3 convert_model.py --input plant_disease_model.tflite --output plant_disease_model.onnx

  # ONNX 转 OM (需在Atlas板上运行, 需CANN环境)
  python3 convert_model.py --input plant_disease_model.onnx --output plant_disease_model.om --soc Ascend310B1

  # 直接生成演示用ONNX模型 (无需TFLite源文件)
  python3 convert_model.py --demo --output plant_disease_model.onnx
"""

import argparse
import sys
import os
import numpy as np


def tflite_to_onnx(tflite_path: str, onnx_path: str):
    """将TFLite模型转换为ONNX格式"""
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print("请安装onnx: pip install onnx")
        sys.exit(1)

    try:
        import tf2onnx
    except ImportError:
        print("请安装tf2onnx: pip install tf2onnx")
        print("或使用 --demo 直接生成演示模型")
        sys.exit(1)

    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        print(f"TFLite模型输入: {input_shape}")

        # 使用tf2onnx转换
        cmd = f"python3 -m tf2onnx.convert --tflite {tflite_path} --output {onnx_path} --opset 13"
        print(f"执行转换: {cmd}")
        os.system(cmd)
        print(f"ONNX模型已保存: {onnx_path}")

    except ImportError:
        print("请安装tensorflow: pip install tensorflow")
        sys.exit(1)


def onnx_to_om(onnx_path: str, om_path: str, soc: str = "Ascend310B1"):
    """将ONNX模型转换为OM格式 (需在Atlas板上运行, 需ATC工具)"""
    atc_cmd = (
        f"atc "
        f"--model={onnx_path} "
        f"--output={om_path} "
        f"--framework=5 "  # 5 = ONNX
        f"--soc={soc} "
        f"--input_shape=\"input:1,3,96,96\" "
        f"--output_type=fp32"
    )
    print(f"执行ATC转换 (需CANN环境):")
    print(f"  {atc_cmd}")
    ret = os.system(atc_cmd)
    if ret == 0:
        print(f"OM模型已保存: {om_path}")
    else:
        print(f"ATC转换失败 (退出码: {ret})")
        print("请确保:")
        print("  1. 在Atlas 200I DK A2板上运行")
        print("  2. CANN SDK已安装并source环境变量")
        print("  3. atc命令可用")
    return ret == 0


def create_demo_onnx(onnx_path: str):
    """
    生成演示用草莓病害检测ONNX模型
    5类: 健康/炭疽病/灰霉病/叶灼病/白粉病
    输入: 1x3x96x96 float32 (NCHW, 归一化到[-1,1])
    输出: 1x5 float32 (softmax概率)
    """
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        print("请安装onnx: pip install onnx")
        sys.exit(1)

    print("正在生成演示用CNN模型 (5类草莓病害检测)...")

    # 定义一个轻量CNN: Conv3x3->BN->ReLU->Pool -> Conv3x3->BN->ReLU->Pool -> FC->FC
    # 输入: 1x3x96x96
    nodes = []
    initializers = []
    graph_inputs = []
    graph_outputs = []

    def make_conv(name, in_ch, out_ch, ksize=3, pad=1):
        weight = numpy_helper.from_array(
            (np.random.randn(out_ch, in_ch, ksize, ksize).astype(np.float32) * 0.05),
            name=f"{name}_w"
        )
        bias = numpy_helper.from_array(
            np.zeros(out_ch, dtype=np.float32), name=f"{name}_b"
        )
        initializers.extend([weight, bias])
        return weight, bias

    def make_bn(name, channels):
        scale = numpy_helper.from_array(np.ones(channels, dtype=np.float32), name=f"{name}_scale")
        bias = numpy_helper.from_array(np.zeros(channels, dtype=np.float32), name=f"{name}_bias")
        mean = numpy_helper.from_array(np.zeros(channels, dtype=np.float32), name=f"{name}_mean")
        var = numpy_helper.from_array(np.ones(channels, dtype=np.float32), name=f"{name}_var")
        initializers.extend([scale, bias, mean, var])
        return scale, bias, mean, var

    # Input
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 96, 96])
    graph_inputs.append(input_tensor)

    # Conv1: 3->16, 3x3, pad=1 -> 96x96
    w1, b1 = make_conv('conv1', 3, 16)
    nodes.append(helper.make_node('Conv', ['input', 'conv1_w', 'conv1_b'], ['conv1_out'],
                                   kernel_shape=[3,3], pads=[1,1,1,1]))
    bn1_s, bn1_b, bn1_m, bn1_v = make_bn('bn1', 16)
    nodes.append(helper.make_node('BatchNormalization',
                                   ['conv1_out', 'bn1_scale', 'bn1_bias', 'bn1_mean', 'bn1_var'],
                                   ['bn1_out'], epsilon=1e-5))
    nodes.append(helper.make_node('Relu', ['bn1_out'], ['relu1_out']))
    nodes.append(helper.make_node('MaxPool', ['relu1_out'], ['pool1_out'],
                                   kernel_shape=[2,2], strides=[2,2]))  # -> 48x48

    # Conv2: 16->32, 3x3, pad=1 -> 48x48
    w2, b2 = make_conv('conv2', 16, 32)
    nodes.append(helper.make_node('Conv', ['pool1_out', 'conv2_w', 'conv2_b'], ['conv2_out'],
                                   kernel_shape=[3,3], pads=[1,1,1,1]))
    bn2_s, bn2_b, bn2_m, bn2_v = make_bn('bn2', 32)
    nodes.append(helper.make_node('BatchNormalization',
                                   ['conv2_out', 'bn2_scale', 'bn2_bias', 'bn2_mean', 'bn2_var'],
                                   ['bn2_out'], epsilon=1e-5))
    nodes.append(helper.make_node('Relu', ['bn2_out'], ['relu2_out']))
    nodes.append(helper.make_node('MaxPool', ['relu2_out'], ['pool2_out'],
                                   kernel_shape=[2,2], strides=[2,2]))  # -> 24x24

    # Conv3: 32->64, 3x3, pad=1 -> 24x24
    w3, b3 = make_conv('conv3', 32, 64)
    nodes.append(helper.make_node('Conv', ['pool2_out', 'conv3_w', 'conv3_b'], ['conv3_out'],
                                   kernel_shape=[3,3], pads=[1,1,1,1]))
    bn3_s, bn3_b, bn3_m, bn3_v = make_bn('bn3', 64)
    nodes.append(helper.make_node('BatchNormalization',
                                   ['conv3_out', 'bn3_scale', 'bn3_bias', 'bn3_mean', 'bn3_var'],
                                   ['bn3_out'], epsilon=1e-5))
    nodes.append(helper.make_node('Relu', ['bn3_out'], ['relu3_out']))
    nodes.append(helper.make_node('MaxPool', ['relu3_out'], ['pool3_out'],
                                   kernel_shape=[2,2], strides=[2,2]))  # -> 12x12

    # Global Average Pool -> 64x1x1
    nodes.append(helper.make_node('GlobalAveragePool', ['pool3_out'], ['gap_out']))

    # Flatten
    nodes.append(helper.make_node('Flatten', ['gap_out'], ['flat_out'], axis=1))

    # FC1: 64 -> 32
    fc1_w = numpy_helper.from_array(
        (np.random.randn(64, 32).astype(np.float32) * 0.1), name='fc1_w')
    fc1_b = numpy_helper.from_array(np.zeros(32, dtype=np.float32), name='fc1_b')
    initializers.extend([fc1_w, fc1_b])
    nodes.append(helper.make_node('MatMul', ['flat_out', 'fc1_w'], ['fc1_matmul']))
    nodes.append(helper.make_node('Add', ['fc1_matmul', 'fc1_b'], ['fc1_out']))
    nodes.append(helper.make_node('Relu', ['fc1_out'], ['fc1_relu']))

    # FC2: 32 -> 5
    fc2_w = numpy_helper.from_array(
        (np.random.randn(32, 5).astype(np.float32) * 0.1), name='fc2_w')
    fc2_b = numpy_helper.from_array(
        np.array([2.0, -0.5, -0.5, -0.5, -0.5], dtype=np.float32),  # 偏向健康类
        name='fc2_b')
    initializers.extend([fc2_w, fc2_b])
    nodes.append(helper.make_node('MatMul', ['fc1_relu', 'fc2_w'], ['fc2_matmul']))
    nodes.append(helper.make_node('Add', ['fc2_matmul', 'fc2_b'], ['fc2_out']))

    # Softmax
    nodes.append(helper.make_node('Softmax', ['fc2_out'], ['output'], axis=1))

    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 5])
    graph_outputs.append(output_tensor)

    graph = helper.make_graph(nodes, 'plant_disease_demo', graph_inputs, graph_outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    print(f"演示ONNX模型已保存: {onnx_path}")
    print(f"  输入: input [1,3,96,96] float32 (NCHW, 范围[-1,1])")
    print(f"  输出: output [1,5] float32 (softmax: 健康/炭疽病/灰霉病/叶灼病/白粉病)")
    print(f"  注意: 这是随机初始化权重的演示模型, 检测结果无实际意义")
    print(f"        生产环境请使用训练好的TFLite/ONNX模型通过 --input 转换")


def main():
    parser = argparse.ArgumentParser(description='植物病害检测模型转换工具')
    parser.add_argument('--input', help='输入模型路径 (.tflite 或 .onnx)')
    parser.add_argument('--output', required=True, help='输出模型路径 (.onnx 或 .om)')
    parser.add_argument('--soc', default='Ascend310B1', help='ATC目标SoC (默认: Ascend310B1)')
    parser.add_argument('--demo', action='store_true', help='生成演示用ONNX模型 (随机权重)')
    args = parser.parse_args()

    if args.demo:
        create_demo_onnx(args.output)
        return

    if not args.input:
        print("错误: 需要 --input 或 --demo")
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)

    if args.input.endswith('.tflite') and args.output.endswith('.onnx'):
        tflite_to_onnx(args.input, args.output)
    elif args.input.endswith('.onnx') and args.output.endswith('.om'):
        onnx_to_om(args.input, args.output, args.soc)
    else:
        print(f"不支持的转换: {args.input} -> {args.output}")
        print("支持的转换:")
        print("  .tflite -> .onnx  (TFLite转ONNX)")
        print("  .onnx -> .om     (ONNX转OM, 需ATC)")
        sys.exit(1)


if __name__ == '__main__':
    main()
