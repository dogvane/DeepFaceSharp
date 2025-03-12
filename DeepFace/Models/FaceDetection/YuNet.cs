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
    public class YuNet : IDisposable, IDetection
    {
        private readonly InferenceSession _session;
        public class Config : ModelBaseConfig
        {
            public float ScoreThreshold { get; set; } = 0.9f;

            public Config()
            {
                var path = ModelConfiguration.Instance.ModelsDirectory;
                ModelFile = Path.Combine(path, "yunet.onnx");
                ModelUrl = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx";
                DeviceId = ModelConfiguration.Instance.DeviceId;
                PreferredBackend = ModelConfiguration.Instance.PreferredGPUBackend;
                ScoreThreshold = ModelConfiguration.Instance.DetectionThreshold;
            }
        }

        private Config _config;

        public YuNet() : this(new Config())
        {

        }

        public YuNet(Config config)
        {
            _config = config;

            // 检查模型文件是否存在，不存在则尝试下载
            ModelDownloadUtils.EnsureModelExists(_config.ModelFile, _config.ModelUrl);
            _session = OnnxUtils.CreateSession(_config.ModelFile, _config.DeviceId, _config.PreferredBackend);
        }

        public List<DetectionResult> DetectFaces(Mat image)
        {
            var results = new List<DetectionResult>();
            
            // 记录原始尺寸信息
            var originalWidth = image.Width;
            var originalHeight = image.Height;
            
            var resizeResult = image.Resize(640, 640);
            var processedImage = resizeResult.Image;

            var tensor = ImageUtils.ImgToTensortPY(processedImage, normalize:false);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };

            using (var inferenceResults = _session.Run(inputs))
            {
                var strides = new[] { 8, 16, 32 };
                var padW = (int)(((640.0f - 1) / 32) + 1) * 32;
                var padH = (int)(((640.0f - 1) / 32) + 1) * 32;

                foreach (var stride in strides)
                {
                    var cols = padW / stride;
                    var rows = padH / stride;

                    var cls = inferenceResults.FirstOrDefault(x => x.Name == $"cls_{stride}")?.AsEnumerable<float>().ToArray();
                    var obj = inferenceResults.FirstOrDefault(x => x.Name == $"obj_{stride}")?.AsEnumerable<float>().ToArray();
                    var bbox = inferenceResults.FirstOrDefault(x => x.Name == $"bbox_{stride}")?.AsEnumerable<float>().ToArray();
                    var kps = inferenceResults.FirstOrDefault(x => x.Name == $"kps_{stride}")?.AsEnumerable<float>().ToArray();

                    if (cls == null || obj == null || bbox == null || kps == null) continue;
                    var maxScore = float.MinValue;

                    for (int row = 0; row < rows; row++)
                    {
                        for (int col = 0; col < cols; col++)
                        {
                            var idx = row * cols + col;

                            // 计算得分
                            float clsScore = Math.Min(Math.Max(cls[idx], 0.0f), 1.0f);
                            float objScore = Math.Min(Math.Max(obj[idx], 0.0f), 1.0f);

                            float score = (float)Math.Sqrt(clsScore * objScore);
                            maxScore = Math.Max(maxScore, score);

                            if (score < _config.ScoreThreshold)
                                continue;

                            // 计算边界框
                            float cx = ((col + bbox[idx * 4 + 0]) * stride);
                            float cy = ((row + bbox[idx * 4 + 1]) * stride);
                            float width = (float)(Math.Exp(bbox[idx * 4 + 2]) * stride);
                            float height = (float)(Math.Exp(bbox[idx * 4 + 3]) * stride);

                            float x = cx - width / 2.0f;
                            float y = cy - height / 2.0f;

                            // 创建检测结果 - 使用640x640尺度的坐标
                            var result = new DetectionResult
                            {
                                FacialArea = new FacialArea
                                {
                                    X = (int)x,
                                    Y = (int)y,
                                    W = (int)width,
                                    H = (int)height,
                                },
                                Confidence = score
                            };

                            // 添加关键点 - 使用640x640尺度的坐标
                            float[] landmarks = new float[10];
                            for (int n = 0; n < 5; n++)
                            {
                                float kx = (kps[idx * 10 + 2 * n] + col) * stride;
                                float ky = (kps[idx * 10 + 2 * n + 1] + row) * stride;
                                landmarks[2 * n] = kx;
                                landmarks[2 * n + 1] = ky;
                            }

                            // 设置关键点
                            result.FacialArea.LeftEye = ((int)landmarks[0], (int)landmarks[1]);
                            result.FacialArea.RightEye = ((int)landmarks[2], (int)landmarks[3]);
                            result.FacialArea.Nose = ((int)landmarks[4], (int)landmarks[5]);
                            result.FacialArea.MouthLeft = ((int)landmarks[6], (int)landmarks[7]);
                            result.FacialArea.MouthRight = ((int)landmarks[8], (int)landmarks[9]);

                            results.Add(result);
                        }
                    }

                    Console.WriteLine($"stride:{stride} maxScore:{maxScore} ");
                }

                // 先进行 NMS
                results = results.ApplyNMS();
            }

            // NMS 之后再进行坐标转换
            if (resizeResult.IsResized)
            {
                foreach (var result in results)
                {
                    // 还原边界框坐标
                    result.FacialArea.X = (int)((result.FacialArea.X - resizeResult.OffsetX) / resizeResult.Ratio);
                    result.FacialArea.Y = (int)((result.FacialArea.Y - resizeResult.OffsetY) / resizeResult.Ratio);
                    result.FacialArea.W = (int)(result.FacialArea.W / resizeResult.Ratio);
                    result.FacialArea.H = (int)(result.FacialArea.H / resizeResult.Ratio);

                    // 还原关键点坐标
                    if (result.FacialArea.LeftEye.HasValue)
                        result.FacialArea.LeftEye = ((int)((result.FacialArea.LeftEye.Value.x - resizeResult.OffsetX) / resizeResult.Ratio), 
                                                   (int)((result.FacialArea.LeftEye.Value.y - resizeResult.OffsetY) / resizeResult.Ratio));
                    if (result.FacialArea.RightEye.HasValue)
                        result.FacialArea.RightEye = ((int)((result.FacialArea.RightEye.Value.x - resizeResult.OffsetX) / resizeResult.Ratio), 
                                                    (int)((result.FacialArea.RightEye.Value.y - resizeResult.OffsetY) / resizeResult.Ratio));
                    if (result.FacialArea.Nose.HasValue)
                        result.FacialArea.Nose = ((int)((result.FacialArea.Nose.Value.x - resizeResult.OffsetX) / resizeResult.Ratio), 
                                                (int)((result.FacialArea.Nose.Value.y - resizeResult.OffsetY) / resizeResult.Ratio));
                    if (result.FacialArea.MouthLeft.HasValue)
                        result.FacialArea.MouthLeft = ((int)((result.FacialArea.MouthLeft.Value.x - resizeResult.OffsetX) / resizeResult.Ratio), 
                                                     (int)((result.FacialArea.MouthLeft.Value.y - resizeResult.OffsetY) / resizeResult.Ratio));
                    if (result.FacialArea.MouthRight.HasValue)
                        result.FacialArea.MouthRight = ((int)((result.FacialArea.MouthRight.Value.x - resizeResult.OffsetX) / resizeResult.Ratio), 
                                                      (int)((result.FacialArea.MouthRight.Value.y - resizeResult.OffsetY) / resizeResult.Ratio));
                }
            }

            return results;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }

        public List<DetectionResult> DetectFaces(string imagePath)
        {
            return DetectFaces(Cv2.ImRead(imagePath));
        }
    }
}
