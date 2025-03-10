using System;
using System.IO;
using DeepFace.Config;
using DeepFace.Factory;
using DeepFace.Models;

namespace DeepFace
{

    /// <summary>
    /// 人脸识别模型后端类型
    /// </summary>
    public enum RepresentationBackend
    {
        ArcFace,
        FaceNet
    }

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

    /// <summary>
    /// 人脸检测类配置类
    /// </summary>
    public class DeepFaceConfig
    {
        /// <summary>
        /// 检测模型后端，默认为ArcFace
        /// </summary>
        public RepresentationBackend RepresentationBackend { get; set; } = RepresentationBackend.ArcFace;

        /// <summary>
        /// 人脸识别后端，默认为YuNet
        /// </summary>
        public DetectorBackend DetectorBackend { get; set; } = DetectorBackend.YuNet;

        /// <summary>
        /// 是否对齐人脸，默认为true
        /// </summary>
        public bool Align { get; set; } = true;

        /// <summary>
        /// 扩展百分比，默认为0
        /// </summary>
        public int ExpandPercentage { get; set; } = 0;
    }


    /// <summary>
    /// DeepFace主类，用于初始化和配置模型
    /// </summary>
    public partial class DeepFace
    {
        private readonly DeepFaceConfig _config;
        private readonly IDetection _detector;
        private readonly IRecognition _recognizer;

        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="config">表示模型配置，如果为null则使用默认配置</param>
        public DeepFace(DeepFaceConfig config = null)
        {
            _config = config ?? new DeepFaceConfig();
            // 在构造函数中初始化检测器和识别器
            _detector = FaceFactory.CreateDetector(_config.DetectorBackend);
            _recognizer = FaceFactory.CreateRecognizer(_config.RepresentationBackend);
        }

    }
}
