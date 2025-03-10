# pylint: disable=unsubscriptable-object

# --------------------------------
# dependency configuration

from tensorflow.keras.models import Model
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import (
    ZeroPadding2D,
    Input,
    Conv2D,
    BatchNormalization,
    PReLU,
    Add,
    Dropout,
    Flatten,
    Dense,
)
import os
import sys


import tensorflow as tf
import onnx
import tf2onnx
import cv2  # 添加opencv导入
import numpy as np  # 添加numpy导入
import onnxruntime  # 新增导入


WEIGHTS_URL="https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5"

# pylint: disable=too-few-public-methods
class ArcFaceClient():
    """
    ArcFace model class
    """

    def __init__(self):
        self.model = load_model()
        self.model_name = "ArcFace"
        self.input_shape = (112, 112)
        self.output_shape = 512

    def export_onnx(self, save_path):
        """
        Export the model to ONNX format
        Args:
            save_path: Path to save the ONNX model
        """
        spec = (tf.TensorSpec((None, 112, 112, 3), tf.float32, name="input"),)
        output_path = os.path.join(save_path, "arcface.onnx")
        
        # Convert to ONNX
        model_proto, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, 
                                                   output_path=output_path)
        print(f"Model exported to: {output_path}")
        return output_path


def load_model(
    url=WEIGHTS_URL,
) -> Model:
    """
    Construct ArcFace model, download its weights and load
    Returns:
        model (Model)
    """
    base_model = ResNet34()
    inputs = base_model.inputs[0]
    arcface_model = base_model.outputs[0]
    arcface_model = BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
    arcface_model = Dropout(0.4)(arcface_model)
    arcface_model = Flatten()(arcface_model)
    arcface_model = Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(
        arcface_model
    )
    embedding = BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(
        arcface_model
    )
    model = Model(inputs, embedding, name=base_model.name)

    weight_file = r'G:\\ai_dotnet\\UserSmartBoard\\Assets\\deepface\\arcface_weights.h5'
    
    model.load_weights(weight_file)

    return model


def ResNet34() -> Model:
    """
    ResNet34 model
    Returns:
        model (Model)
    """
    img_input = Input(shape=(112, 112, 3))

    x = ZeroPadding2D(padding=1, name="conv1_pad")(img_input)
    x = Conv2D(
        64, 3, strides=1, use_bias=False, kernel_initializer="glorot_normal", name="conv1_conv"
    )(x)
    x = BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9, name="conv1_bn")(x)
    x = PReLU(shared_axes=[1, 2], name="conv1_prelu")(x)
    x = stack_fn(x)

    model = training.Model(img_input, x, name="ResNet34")

    return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    bn_axis = 3

    if conv_shortcut:
        shortcut = Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            kernel_initializer="glorot_normal",
            name=name + "_0_conv",
        )(x)
        shortcut = BatchNormalization(
            axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_1_bn")(x)
    x = ZeroPadding2D(padding=1, name=name + "_1_pad")(x)
    x = Conv2D(
        filters,
        3,
        strides=1,
        kernel_initializer="glorot_normal",
        use_bias=False,
        name=name + "_1_conv",
    )(x)
    x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_2_bn")(x)
    x = PReLU(shared_axes=[1, 2], name=name + "_1_prelu")(x)

    x = ZeroPadding2D(padding=1, name=name + "_2_pad")(x)
    x = Conv2D(
        filters,
        kernel_size,
        strides=stride,
        kernel_initializer="glorot_normal",
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_3_bn")(x)

    x = Add(name=name + "_add")([shortcut, x])
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    x = block1(x, filters, stride=stride1, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + "_block" + str(i))
    return x


def stack_fn(x):
    x = stack1(x, 64, 3, name="conv2")
    x = stack1(x, 128, 4, name="conv3")
    x = stack1(x, 256, 6, name="conv4")
    return stack1(x, 512, 3, name="conv5")

def export_onnx():
    try:
        arcface = ArcFaceClient()
        # 导出ONNX模型
        save_path = os.path.dirname(r"G:\\ai_dotnet\\UserSmartBoard\\Assets\\deepface\\arcface.onnx")
        
        # 确保输出目录存在
        os.makedirs(save_path, exist_ok=True)
        
        output_path = arcface.export_onnx(save_path)
        print(f"ONNX导出完成: {output_path}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1)

def process_image_tf_readimg():
    arcface = ArcFaceClient()
    imgFime = r'G:\\ai_dotnet\\UserSmartBoard\\SimpleData\\face001.png'
    img = tf.io.read_file(imgFime)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (112, 112))
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    print(img.shape)
    result = arcface.model.predict(img)
    print(result.shape)
    print(result)
    
    # 保存特征向量到文件
    output_file = r'G:\\ai_dotnet\\UserSmartBoard\\SimpleData\\001.txt'
    with open(output_file, 'w') as f:
        # 将numpy数组转换为字符串并保存
        feature_str = '\n'.join(map(str, result[0]))
        f.write(feature_str)
    print(f"特征向量已保存到: {output_file}")
    
def process_image_cv2():
    """
    使用OpenCV读取图片并进行人脸特征提取
    """
    arcface = ArcFaceClient()
    img_file = r'G:\\ai_dotnet\\UserSmartBoard\\SimpleData\\face001.png'
    
    # 使用OpenCV读取图片
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # 模型推理
    result = arcface.model.predict(img)
    print("OpenCV处理结果shape:", result.shape)
    
    # 保存特征向量到文件
    output_file = r'G:\\ai_dotnet\\UserSmartBoard\\SimpleData\\002.txt'
    with open(output_file, 'w') as f:
        feature_str = '\n'.join(map(str, result[0]))
        f.write(feature_str)
    print(f"OpenCV处理的特征向量已保存到: {output_file}")

def process_image_onnx():
    """
    使用ONNX模型进行人脸特征提取
    """
    # 加载ONNX模型
    onnx_path = r'G:\\ai_dotnet\\UserSmartBoard\\Assets\\deepface\\arcface.onnx'
    session = onnxruntime.InferenceSession(onnx_path)
    
    # 读取图片
    img_file = r'G:\\ai_dotnet\\UserSmartBoard\\SimpleData\\face001.png'
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # ONNX模型推理
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: img})[0]
    print("ONNX处理结果shape:", result.shape)
    
    # 保存特征向量到文件
    output_file = r'G:\\ai_dotnet\\UserSmartBoard\\SimpleData\\003.txt'
    with open(output_file, 'w') as f:
        feature_str = '\n'.join(map(str, result[0]))
        f.write(feature_str)
    print(f"ONNX处理的特征向量已保存到: {output_file}")

def main():
    # process_image_tf_readimg() # 添加对新函数的调用
    # process_image_cv2()  # 添加对新函数的调用
    process_image_onnx()  # 添加新函数调用

if __name__ == "__main__":
    main()
