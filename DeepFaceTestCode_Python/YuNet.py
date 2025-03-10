# 内置依赖
import os
from typing import Any, List, Optional,Tuple
from FacialAreaRegion import FacialAreaRegion

# 第三方依赖
import cv2
import numpy as np

# pylint:disable=line-too-long
WEIGHTS_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


class YuNetClient():
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        构建一个yunet检测器模型
        返回:
            model (Any)
        """

        opencv_version = cv2.__version__.split(".")
        if not len(opencv_version) >= 2:
            raise ValueError(
                f"OpenCv的版本必须包含主版本和次版本，但当前版本为 {opencv_version}"
            )

        opencv_version_major = int(opencv_version[0])
        opencv_version_minor = int(opencv_version[1])

        if (opencv_version_major < 4 or (opencv_version_major == 4 and opencv_version_minor < 8)):
            # 最低要求: https://github.com/opencv/opencv_zoo/issues/172
            raise ValueError(f"YuNet 需要 opencv-python >= 4.8，但当前版本为 {cv2.__version__}")

        # pylint: disable=C0301
        weight_file = r'../DeepFace.Console/bin/Debug/net8.0/models/face_detection_yunet_2023mar.onnx'

        try:
            face_detector = cv2.FaceDetectorYN_create(weight_file, "", (0, 0))
        except Exception as err:
            raise ValueError(
                "调用 opencv.FaceDetectorYN_create 模块时发生异常。"
                + "这是一个可选依赖。"
                + "你可以通过 pip install opencv-contrib-python 安装它。"
            ) from err
        return face_detector

    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        使用yunet检测和对齐人脸

        参数:
            img (np.ndarray): 预加载的numpy数组格式图像

        返回:
            results (List[FacialAreaRegion]): FacialAreaRegion对象列表
        """
        # FaceDetector.detect_faces不支持score_threshold参数
        # 我们可以通过环境变量设置它
        score_threshold = float(os.environ.get("yunet_score_threshold", "0.9"))
        resp = []
        faces = []
        height, width = img.shape[0], img.shape[1]
        # 如果图像太大则调整大小(Yunet有时在大尺寸输入上无法检测人脸)
        # 选择640作为阈值是因为这是Yunet中max_size的默认值
        resized = False
        r = 1  # 调整大小的系数
        if (height > 640 or width > 640):
            r = 640.0 / max(height, width)
            img = cv2.resize(img, (int(width * r), int(height * r)))
            height, width = img.shape[0], img.shape[1]
            resized = True
        
        self.model.setInputSize((width, height))
        self.model.setScoreThreshold(score_threshold)
        _, faces = self.model.detect(img)
        
        print('r:',r, 'width:',width, 'height:',height)
        
        if faces is None:
            return resp
        for face in faces:
            """
            检测输出faces是一个CV_32F类型的二维数组,
            其行是检测到的人脸实例,列是人脸的位置和5个面部特征点。
            每一行的格式如下:
            x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
            x_rcm, y_rcm, x_lcm, y_lcm,
            其中x1, y1, w, h是人脸边界框的左上角坐标、宽度和高度,
            {x, y}_{re, le, nt, rcm, lcm}分别代表右眼、左眼、鼻尖、
            嘴角右端和左端的坐标。
            """
            (x, y, w, h, x_le, y_le, x_re, y_re) = list(map(int, face[:8]))

            # YuNet在认为检测到的人脸部分在框架外时会返回负坐标
            x = max(x, 0)
            y = max(y, 0)
            if resized:
                x, y, w, h = int(x / r), int(y / r), int(w / r), int(h / r)
                x_re, y_re, x_le, y_le = (
                    int(x_re / r),
                    int(y_re / r),
                    int(x_le / r),
                    int(y_le / r),
                )
            confidence = float(face[-1])

            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                confidence=confidence,
                left_eye=(x_re, y_re),
                right_eye=(x_le, y_le),
            )
            resp.append(facial_area)
        return resp


def main():
    # 初始化一个FaceDetector对象
    yunet = YuNetClient()
    # 读取图像
    img = cv2.imread(r"../SampleImages/000000001000.jpg")
    # 检测人脸
    faces = yunet.detect_faces(img)
    # 打印检测到的人脸数量和详细信息
    print(f"检测到{len(faces)}张人脸。")
    
    # 在图像上绘制检测结果
    for face in faces:
        # 绘制人脸边界框
        cv2.rectangle(img, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 0), 2)
        # 绘制左眼
        cv2.circle(img, face.left_eye, 2, (255, 0, 0), 2)
        # 绘制右眼
        cv2.circle(img, face.right_eye, 2, (0, 0, 255), 2)
        # 添加置信度文本
        cv2.putText(img, f"{face.confidence:.2%}", (face.x, face.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存结果图像
    cv2.imwrite("out.jpg", img)
    
    # 打印检测信息
    for i, face in enumerate(faces, 1):
        print(f"\n人脸 #{i}:")
        print(f"位置: x={face.x}, y={face.y}, 宽度={face.w}, 高度={face.h}")
        print(f"置信度: {face.confidence:.2%}")
        print(f"左眼坐标: {face.left_eye}")
        print(f"右眼坐标: {face.right_eye}")

main()
