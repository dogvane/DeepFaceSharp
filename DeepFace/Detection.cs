using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeepFace.Models;
using DeepFace.Config;
using System.Drawing;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace DeepFace
{
    /// <summary>
    /// 人脸检测后端类型
    /// </summary>
    public enum DetectorBackend
    {
        YuNet,
        /// <summary>YOLOv8</summary>
        YoloV8,
        /// <summary>YOLOv11 Nano版本</summary>
        YoloV11n,
        /// <summary>YOLOv11 Small版本</summary>
        YoloV11s,
        /// <summary>YOLOv11 Medium版本</summary>
        YoloV11m,
    }

    public class Detection
    {
        public static List<DetectionResult> ExtractFaces(
            string imgPath,
            DetectorBackend detectorBackend = DetectorBackend.YuNet,
            bool outputFaceMat = true, // 是否输出人脸Mat对象
            bool align = true, // 是否启用人脸对齐(默认为True) 注：只有当outputFaceMat为True时才有效，仅作用于Face对象
            int expandPercentage = 0 // 人脸区域扩展百分比 (0-100) 注：只有当outputFaceMat为True时才有效，仅作用于Face对象
            )
        {
            if (string.IsNullOrEmpty(imgPath) || !File.Exists(imgPath))
            {
                throw new ArgumentException($"无效的图像路径: {imgPath}");
            }

            // 加载图像
            using var sourceImage = Cv2.ImRead(imgPath);
            if (sourceImage.Empty())
            {
                throw new ArgumentException($"无法加载图像: {imgPath}");
            }

            // 获取检测器实例
            var detector = Factory.FaceFactory.CreateDetector(detectorBackend.ToString().ToLower());

            // 检测人脸
            var detectionResults = detector.DetectFacesFromImage(sourceImage);

            if (!detectionResults.Any())
            {
                Console.WriteLine("未检测出人脸!");
                return detectionResults;
            }

            if (outputFaceMat)
            {
                // 使用新的人脸对齐算法处理每个检测结果
                foreach (var result in detectionResults)
                {
                    // 使用FaceAlignment类提取并对齐人脸
                    result.Face = FaceAlignment.ExtractFace(
                        result.FacialArea,
                        sourceImage,
                        align,
                        expandPercentage
                    );
                }
            }

            return detectionResults;
        }
    }
}
