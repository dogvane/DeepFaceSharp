using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using Microsoft.VisualBasic;
using System.IO;
using DeepFace.Common;
using DeepFace.Config;

namespace DeepFace.Models
{
    /// <summary>
    /// 模型仅检测出人脸的区域,包含5个点的关键信息
    /// </summary>
    public class Yolo : IDisposable, IDetection
    {
        private readonly InferenceSession _session;
        
        /// <summary>
        /// 可用的Yolo模型类型
        /// </summary>
        public enum YoloModel
        {
            YoloV8NFace,
            YoloV11NFace,
            YoloV11SFace,
            YoloV11MFace
        }
        
        private static readonly string[] WEIGHT_NAMES = new string[]
        {
            "yolov8n-face.onnx",
            "yolov11n-face.onnx",
            "yolov11s-face.onnx",
            "yolov11m-face.onnx"
        };
        
        // 模型下载地址字典
        private static readonly Dictionary<string, string> MODEL_DOWNLOAD_URLS = new Dictionary<string, string>
        {
            { "yolov8n-face.onnx", "https://github.com/dogvane/DeepFaceSharp/releases/download/v0.0.0/yolov8n-face.onnx" },
            { "yolov11n-face.onnx", "https://github.com/dogvane/DeepFaceSharp/releases/download/v0.0.0/yolov11n-face.onnx" },
            { "yolov11s-face.onnx", "https://github.com/dogvane/DeepFaceSharp/releases/download/v0.0.0/yolov11s-face.onnx" },
            { "yolov11m-face.onnx", "https://github.com/dogvane/DeepFaceSharp/releases/download/v0.0.0/yolov11m-face.onnx" }
        };

        public class Config
        {
            public string ModelPath { get; set; } = "models/yolov8n-face.onnx";
            public float ScoreThreshold { get; set; } = 0.9f;
            public YoloModel ModelType { get; set; } = YoloModel.YoloV8NFace;
            
            public Config() { }
            
            public Config(YoloModel modelType)
            {
                ModelType = modelType;
                ModelPath = $"models/{WEIGHT_NAMES[(int)modelType]}";
            }
        }

        private Config _config;

        public Yolo() : this(YoloModel.YoloV8NFace)
        {

        }
        public Yolo(YoloModel modelType = YoloModel.YoloV8NFace)
        {
            _config = new Config
            {
                ModelType = modelType,
                ModelPath = ModelConfiguration.Instance.GetModelPath(WEIGHT_NAMES[(int)modelType]),
                ScoreThreshold = ModelConfiguration.Instance.DetectionThreshold
            };

            EnsureModelExists(_config.ModelPath, WEIGHT_NAMES[(int)modelType]);

            if (File.Exists(_config.ModelPath))
            {
                _session = new InferenceSession(_config.ModelPath);
                _session.PrintOnnxMetadata();
            }
            else
            {
                // 模型文件未找到，抛出文件未找到异常
                throw new FileNotFoundException($"Model file not found: {_config.ModelPath}\nPlease make sure model configuration is initialized and the file exists.");
            }
        }

        public Yolo(Config config = null)
        {
            if (config == null)
            {
                if (!ModelConfiguration.Instance.IsInitialized)
                {
                    _config = new Config();
                }
                else
                {
                    YoloModel modelType = YoloModel.YoloV8NFace; // 默认使用YoloV8NFace
                    
                    _config = new Config
                    {
                        ModelType = modelType,
                        ModelPath = ModelConfiguration.Instance.GetModelPath(WEIGHT_NAMES[(int)modelType]),
                        ScoreThreshold = ModelConfiguration.Instance.DetectionThreshold
                    };
                }
            }
            else
            {
                _config = config;
            }
            
            string modelFileName = Path.GetFileName(_config.ModelPath);
            EnsureModelExists(_config.ModelPath, modelFileName);
            
            if (File.Exists(_config.ModelPath))
            {
                _session = new InferenceSession(_config.ModelPath);
                _session.PrintOnnxMetadata();
            }
            else
            {
                // 模型文件未找到，抛出文件未找到异常
                throw new FileNotFoundException($"Model file not found: {_config.ModelPath}\nPlease make sure model configuration is initialized and the file exists.");
            }
        }
        
        /// <summary>
        /// 确保模型文件存在，如果不存在则从GitHub下载
        /// </summary>
        /// <param name="modelPath">模型本地路径</param>
        /// <param name="modelFileName">模型文件名</param>
        private void EnsureModelExists(string modelPath, string modelFileName)
        {
            if (!File.Exists(modelPath) && MODEL_DOWNLOAD_URLS.ContainsKey(modelFileName))
            {
                // 提示模型文件不存在，开始下载
                Console.WriteLine($"Model file does not exist: {modelPath}, downloading from GitHub...");
                
                // 确保目录存在
                string directory = Path.GetDirectoryName(modelPath);
                if (!Directory.Exists(directory))
                {
                    Directory.CreateDirectory(directory);
                }
                
                try
                {
                    // 使用ModelDownloadUtils下载模型
                    ModelDownloadUtils.EnsureModelExists(
                        modelPath,
                        MODEL_DOWNLOAD_URLS[modelFileName],
                        ModelDownloadUtils.CreateDefaultProgressHandler()
                    );
                    
                    // 提示模型文件下载成功
                    Console.WriteLine($"Model file downloaded successfully: {modelPath}");
                }
                catch (Exception ex)
                {
                    // 提示模型文件下载失败
                    Console.WriteLine($"Model file download failed: {ex.Message}");
                    // 下载失败仍会继续，让后续的File.Exists检查决定是否抛出异常
                }
            }
        }

        public List<DetectionResult> DetectFaces(Mat image)
        {
            var results = new List<DetectionResult>();
            
            // 定义模型输入大小
            int inputWidth = 640;
            int inputHeight = 640;

            // 调整图像大小，保持纵横比
            var resizeResult = image.Resize(inputWidth, inputHeight);

            // 准备输入张量 (使用标准化处理，将图像从BGR格式转换为RGB格式)
            var inputTensor = ImageUtils.ImgToTensortPY(resizeResult.Image, false);
            var inputName = _session.InputMetadata.Keys.First();
            
            // 创建输入数据
            var inputs = new List<NamedOnnxValue> { 
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor) 
            };

            // 运行推理
            var outputs = _session.Run(inputs);

            // 获取输出数据
            var outputData = outputs.First().AsTensor<float>();

            // 获取输出维度
            var dimensions = outputData.Dimensions.ToArray();
            
            // 解析模型输出 [1, x, y] 形状，取决于模型类型
            // 对于YOLOv8-face通常是 [1, 20, 8400]，YOLOv11-face是 [1, 5, 8400]
            
            List<DetectionResult> detectionResults = new List<DetectionResult>();
            
            int rows = dimensions[1];
            int cols = dimensions[2];
            
            bool isYoloV11 = _config.ModelType == YoloModel.YoloV11NFace || 
                             _config.ModelType == YoloModel.YoloV11SFace ||
                             _config.ModelType == YoloModel.YoloV11MFace;
            
            // 创建列表存储有效的检测结果
            var boxes = new List<float[]>();
            var confidences = new List<float>();
            var keypoints = new List<(int, int)?[]>();
            
            // 遍历所有预测框
            for (int i = 0; i < cols; i++)
            {
                float confidence;
                float x_center, y_center, width, height;
                
                if (isYoloV11)
                {
                    // YoloV11 只有5个值 [x, y, w, h, conf]
                    confidence = outputData[0, 4, i];
                    
                    // 过滤低置信度的检测结果
                    if (confidence < _config.ScoreThreshold) continue;
                    
                    x_center = outputData[0, 0, i];
                    y_center = outputData[0, 1, i];
                    width = outputData[0, 2, i];
                    height = outputData[0, 3, i];
                    
                    // 将边界框转换到原始图像尺寸，考虑缩放比例和偏移
                    float mappedX = (x_center - resizeResult.OffsetX) / resizeResult.Ratio;
                    float mappedY = (y_center - resizeResult.OffsetY) / resizeResult.Ratio;
                    float mappedWidth = width / resizeResult.Ratio;
                    float mappedHeight = height / resizeResult.Ratio;
                    
                    int x1 = (int)Math.Max(0, (mappedX - mappedWidth / 2));
                    int y1 = (int)Math.Max(0, (mappedY - mappedHeight / 2));
                    int w = (int)mappedWidth;
                    int h = (int)mappedHeight;
                    
                    // YoloV11 没有关键点数据，所以初始化为空
                    var kpts = new (int, int)?[5];
                    
                    boxes.Add(new float[] { x1, y1, w, h });
                    confidences.Add(confidence);
                    keypoints.Add(kpts);
                }
                else
                {
                    // YoloV8 有20个值 [x, y, w, h, conf, 5个关键点(每个3个值)]
                    confidence = outputData[0, 4, i];
                    
                    // 过滤低置信度的检测结果
                    if (confidence < _config.ScoreThreshold) continue;
                    
                    x_center = outputData[0, 0, i];
                    y_center = outputData[0, 1, i];
                    width = outputData[0, 2, i];
                    height = outputData[0, 3, i];
                    
                    // 将边界框转换到原始图像尺寸，考虑缩放比例和偏移
                    float mappedX = (x_center - resizeResult.OffsetX) / resizeResult.Ratio;
                    float mappedY = (y_center - resizeResult.OffsetY) / resizeResult.Ratio;
                    float mappedWidth = width / resizeResult.Ratio;
                    float mappedHeight = height / resizeResult.Ratio;
                    
                    int x1 = (int)Math.Max(0, (mappedX - mappedWidth / 2));
                    int y1 = (int)Math.Max(0, (mappedY - mappedHeight / 2));
                    int w = (int)mappedWidth;
                    int h = (int)mappedHeight;
                    
                    // 提取5个关键点 (每个关键点有x,y,conf三个值)，并应用缩放和偏移
                    var kpts = new (int, int)?[5];
                    for (int j = 0; j < 5; j++)
                    {
                        float kp_x = outputData[0, 5 + j * 3, i];
                        float kp_y = outputData[0, 6 + j * 3, i];
                        float kp_conf = outputData[0, 7 + j * 3, i];
                        
                        // 将关键点坐标从输入图像尺寸映射回原始图像尺寸
                        float mappedKpX = (kp_x - resizeResult.OffsetX) / resizeResult.Ratio;
                        float mappedKpY = (kp_y - resizeResult.OffsetY) / resizeResult.Ratio;
                        
                        // 如果关键点置信度足够高，则添加关键点
                        if (kp_conf > 0.5)
                        {
                            kpts[j] = ((int)mappedKpX, (int)mappedKpY);
                        }
                        else
                        {
                            kpts[j] = null;
                        }
                    }
                    
                    boxes.Add(new float[] { x1, y1, w, h });
                    confidences.Add(confidence);
                    keypoints.Add(kpts);
                }
            }
            
            // 创建检测结果
            for (int i = 0; i < boxes.Count; i++)
            {
                var box = boxes[i];
                var confidence = confidences[i];
                var kpts = keypoints[i];
                
                var facial = new FacialArea
                {
                    X = (int)box[0],
                    Y = (int)box[1],
                    W = (int)box[2],
                    H = (int)box[3],
                    LeftEye = kpts[1],      // 左眼索引为1
                    RightEye = kpts[0],     // 右眼索引为0
                    Nose = kpts[2],         // 鼻子索引为2
                    MouthLeft = kpts[3],    // 左嘴角索引为3
                    MouthRight = kpts[4]    // 右嘴角索引为4
                };
                
                var result = new DetectionResult
                {
                    FacialArea = facial,
                    Confidence = confidence
                };
                
                detectionResults.Add(result);
            }
            
            // 应用非极大值抑制 (NMS)
            detectionResults = detectionResults.ApplyNMS(0.4f);
            
            return detectionResults;
        }

        // 添加Sigmoid函数
        private float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        public void Dispose()
        {
            _session?.Dispose();
        }

        public List<DetectionResult> DetectFaces(string imagePath)
        {
            return DetectFaces(Cv2.ImRead(imagePath));
        }

        public List<DetectionResult> DetectFacesFromImage(Mat image)
        {
            return DetectFaces(image);
        }
    }
}
