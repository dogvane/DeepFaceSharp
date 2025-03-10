# built-in dependencies
import os
from typing import List, Any
from enum import Enum
from FacialAreaRegion import FacialAreaRegion

# 3rd party dependencies
import numpy as np
import cv2

class YoloModel(Enum):
    V8N = 0
    V11N = 1
    V11S = 2
    V11M = 3


# Model's weights paths
WEIGHT_NAMES = ["yolov8n-face.pt",
                "yolov11n-face.pt",
                "yolov11s-face.pt",
                "yolov11m-face.pt"]

# Google Drive URL from repo (https://github.com/derronqi/yolov8-face) ~6MB
WEIGHT_URLS = ["https://drive.google.com/uc?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11s-face.pt",
               "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11m-face.pt"]


class YoloDetectorClient():
    def __init__(self, model: YoloModel):
        super().__init__()
        self.model = self.build_model(model)

    def build_model(self, model: YoloModel) -> Any:
        """
        Build a yolo detector model
        Returns:
            model (Any)
        """

        # Import the optional Ultralytics YOLO model
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as e:
            raise ImportError(
                "Yolo is an optional detector, ensure the library is installed. "
                "Please install using 'pip install ultralytics'"
            ) from e

        weight_file = r'G:/ai_dotnet/UserSmartBoard/Assets/deepface' + '/' + WEIGHT_NAMES[model.value]

        # Return face_detector
        return YOLO(weight_file)

    def export_onnx(self, out_path: str):
        self.model.export(format="onnx")
        
    def detect_faces(self, img: np.ndarray) -> List[FacialAreaRegion]:
        """
        Detect and align face with yolo

        Args:
            img (np.ndarray): pre-loaded image as numpy array

        Returns:
            results (List[FacialAreaRegion]): A list of FacialAreaRegion objects
        """
        resp = []

        # Detect faces
        results = self.model.predict(
            img,
            verbose=False,
            show=False,
            conf=float(os.getenv("YOLO_MIN_DETECTION_CONFIDENCE", "0.25")),
        )[0]

        # For each face, extract the bounding box, the landmarks and confidence
        for result in results:

            if result.boxes is None:
                continue

            # Extract the bounding box and the confidence
            x, y, w, h = result.boxes.xywh.tolist()[0]
            confidence = result.boxes.conf.tolist()[0]

            right_eye = None
            left_eye = None

            # yolo-facev8 is detecting eyes through keypoints,
            # while for v11 keypoints are always None
            if result.keypoints is not None:
                # right_eye_conf = result.keypoints.conf[0][0]
                # left_eye_conf = result.keypoints.conf[0][1]
                right_eye = result.keypoints.xy[0][0].tolist()
                left_eye = result.keypoints.xy[0][1].tolist()

                # eyes are list of float, need to cast them tuple of int
                left_eye = tuple(int(i) for i in left_eye)
                right_eye = tuple(int(i) for i in right_eye)

            x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
            facial_area = FacialAreaRegion(
                x=x,
                y=y,
                w=w,
                h=h,
                left_eye=left_eye,
                right_eye=right_eye,
                confidence=confidence,
            )
            resp.append(facial_area)

        return resp


class YoloDetectorClientV8n(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V8N)


class YoloDetectorClientV11n(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V11N)


class YoloDetectorClientV11s(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V11S)


class YoloDetectorClientV11m(YoloDetectorClient):
    def __init__(self):
        super().__init__(YoloModel.V11M)



def main():
    # 初始化一个FaceDetector对象
    yolo = YoloDetectorClientV8n()
    # 读取图像
    img = cv2.imread(r"G:\\ai_dotnet\\UserSmartBoard\\SimpleData\\111.jpg")
    # 检测人脸
    faces = yolo.detect_faces(img)
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


# main()

# YoloDetectorClientV8n().export_onnx("yolov8n-face.onnx")
# YoloDetectorClientV11m().export_onnx("yolov11m-face.onnx")
# YoloDetectorClientV11n().export_onnx("yolov11n-face.onnx")
# YoloDetectorClientV11s().export_onnx("yolov11s-face.onnx")


def dectect_by_onnx():
    import onnxruntime as ort
    import numpy as np
    import cv2
    from FacialAreaRegion import FacialAreaRegion

    weight_file = r'G:/ai_dotnet/UserSmartBoard/Assets/deepface' + '/' + "yolov8n-face.onnx"
    # Load the ONNX model
    model = ort.InferenceSession(weight_file)

    # Load the image
    img_path = "G:\\ai_dotnet\\UserSmartBoard\\SimpleData\\111.jpg"
    img = cv2.imread(img_path)
    orig_img = img.copy()  # 保存原始图像用于绘制
    orig_h, orig_w = img.shape[:2]
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    input_size = 640
    img = cv2.resize(img, (input_size, input_size))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0

    # Run the model
    outputs = model.run(None, {"images": img})
    
    # 解析模型输出 [1, 20, 8400]
    predictions = outputs[0]
    
    # 重新排列输出形状为 [8400, 20]
    predictions = predictions[0].T
    
    # 设置置信度阈值
    conf_threshold = 0.5
    
    # 创建列表存储有效的检测结果
    boxes = []
    confidences = []
    keypoints = []
    
    # 遍历所有预测框
    for i in range(predictions.shape[0]):
        # 提取置信度
        confidence = predictions[i][4]
        
        # 过滤低置信度的检测结果
        if confidence >= conf_threshold:
            # 提取边界框参数
            x_center, y_center, width, height = predictions[i][0:4]
            
            # 将边界框转换到原始图像尺寸
            x1 = int((x_center - width/2) * orig_w / input_size)
            y1 = int((y_center - height/2) * orig_h / input_size)
            w = int(width * orig_w / input_size)
            h = int(height * orig_h / input_size)
            
            # 确保边界框在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            
            # 提取5个关键点
            kpts = []
            for j in range(5):
                kp_x = predictions[i][5 + j*3] * orig_w / input_size
                kp_y = predictions[i][6 + j*3] * orig_h / input_size
                kp_conf = predictions[i][7 + j*3]
                kpts.append((int(kp_x), int(kp_y), kp_conf))
            
            boxes.append([x1, y1, w, h])
            confidences.append(float(confidence))
            keypoints.append(kpts)
    
    # 应用非极大值抑制 (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
    
    # 创建面部区域对象列表
    face_regions = []
    
    # 遍历保留的检测结果
    for i in indices:
        box = boxes[i]
        conf = confidences[i]
        kpts = keypoints[i]
        
        # 提取坐标
        x, y, w, h = box
        
        # 提取关键点坐标 (YOLOv8-face模型中0和1为左右眼)
        right_eye = (kpts[0][0], kpts[0][1])
        left_eye = (kpts[1][0], kpts[1][1])
        
        # 创建面部区域对象
        facial_area = FacialAreaRegion(
            x=x,
            y=y,
            w=w,
            h=h,
            left_eye=left_eye,
            right_eye=right_eye,
            confidence=conf
        )
        face_regions.append(facial_area)
    
    print(f"ONNX模型检测到{len(face_regions)}张人脸")
    
    # 在原图上绘制结果
    for face in face_regions:
        # 绘制边界框
        cv2.rectangle(orig_img, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 0), 2)
        
        # 绘制所有关键点
        for idx, i in enumerate(indices):
            kpts = keypoints[i]
            # 分别绘制五个关键点(右眼、左眼、鼻子、右嘴角、左嘴角)
            colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255)]
            for j, (x, y, conf) in enumerate(kpts):
                if conf > 0.5:  # 仅绘制置信度高的关键点
                    cv2.circle(orig_img, (int(x), int(y)), 3, colors[j], -1)
        
        # 添加置信度文本
        cv2.putText(orig_img, f"{face.confidence:.2f}", (face.x, face.y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 保存结果
    cv2.imwrite("onnx_detection_result.jpg", orig_img)
    
    # 输出检测到的人脸信息
    for i, face in enumerate(face_regions):
        print(f"\n人脸 #{i+1}:")
        print(f"位置: x={face.x}, y={face.y}, 宽度={face.w}, 高度={face.h}")
        print(f"置信度: {face.confidence:.2f}")
        print(f"左眼坐标: {face.left_eye}")
        print(f"右眼坐标: {face.right_eye}")
    
    return face_regions


if __name__ == "__main__":
    dectect_by_onnx()

