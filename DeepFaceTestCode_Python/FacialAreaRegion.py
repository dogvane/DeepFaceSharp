from typing import Any, List, Optional,Tuple


class FacialAreaRegion:
    """
    初始化Face对象。

    参数:
        x (int): 边界框左上角的x坐标。
        y (int): 边界框左上角的y坐标。
        w (int): 边界框的宽度。
        h (int): 边界框的高度。
        left_eye (tuple): 相对于人物而非观察者的左眼坐标(x, y)。
            默认为None。
        right_eye (tuple): 相对于人物而非观察者的右眼坐标(x, y)。
            默认为None。
        confidence (float, optional): 与人脸检测相关的置信度分数。
            默认为None。
        nose (tuple, optional): 鼻子的坐标(x, y)。
            默认为None。
        mouth_right (tuple, optional): 嘴巴右侧的坐标(x, y)。
            默认为None。
        mouth_left (tuple, optional): 嘴巴左侧的坐标(x, y)。
            默认为None。
    """

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        left_eye: Optional[Tuple[int, int]] = None,
        right_eye: Optional[Tuple[int, int]] = None,
        confidence: Optional[float] = None,
        nose: Optional[Tuple[int, int]] = None,
        mouth_right: Optional[Tuple[int, int]] = None,
        mouth_left: Optional[Tuple[int, int]] = None
    ):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.confidence = confidence
        self.nose = nose
        self.mouth_right = mouth_right
        self.mouth_left = mouth_left

    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None
    nose: Optional[Tuple[int, int]] = None
    mouth_right: Optional[Tuple[int, int]] = None
    mouth_left: Optional[Tuple[int, int]] = None

