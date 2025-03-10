using System;
using System.IO;

namespace DeepFace.Config
{
    /// <summary>
    /// 深度学习模型配置类，用于全局管理模型路径
    /// </summary>
    public class ModelConfiguration
    {
        private static ModelConfiguration _instance;
        private static readonly object _lock = new object();

        /// <summary>
        /// 获取全局配置实例
        /// </summary>
        public static ModelConfiguration Instance
        {
            get
            {
                if (_instance == null)
                {
                    lock (_lock)
                    {
                        if (_instance == null)
                        {
                            _instance = new ModelConfiguration();
                        }
                    }
                }
                return _instance;
            }
        }

        /// <summary>
        /// 模型根目录路径
        /// </summary>
        public string ModelsDirectory { get; private set; }

        /// <summary>
        /// 是否已经初始化
        /// </summary>
        public bool IsInitialized { get; private set; }

        /// <summary>
        /// 人脸检测的默认置信度阈值
        /// </summary>
        public float DetectionThreshold { get; private set; };

        private ModelConfiguration()
        {
            ModelsDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models");
            IsInitialized = false;
            DetectionThreshold = 0.9f;
        }

        /// <summary>
        /// 初始化模型配置
        /// </summary>
        /// <param name="modelsDirectory">模型目录路径</param>
        public void Initialize(string modelsDirectory)
        {
            if (IsInitialized)
            {
                throw new InvalidOperationException("Model configuration has already been initialized and cannot be initialized again"); // 模型配置已经初始化过，不能重复初始化
            }

            if (string.IsNullOrEmpty(modelsDirectory))
            {
                throw new ArgumentException("Model directory path cannot be empty"); // 模型目录路径不能为空
            }

            if (!Directory.Exists(modelsDirectory))
            {
                Directory.CreateDirectory(modelsDirectory);
            }

            ModelsDirectory = modelsDirectory;
            IsInitialized = true;
        }

        /// <summary>
        /// 获取完整的模型文件路径
        /// </summary>
        /// <param name="modelFileName">模型文件名</param>
        /// <returns>完整的模型文件路径</returns>
        public string GetModelPath(string modelFileName)
        {
            return Path.Combine(ModelsDirectory, modelFileName);
        }

        /// <summary>
        /// 设置人脸检测的默认置信度阈值
        /// </summary>
        /// <param name="threshold">阈值(0-1之间的浮点数)</param>
        public void SetDetectionThreshold(float threshold)
        {
            if (threshold < 0 || threshold > 1)
            {
                throw new ArgumentOutOfRangeException(nameof(threshold), "Threshold must be between 0 and 1"); // 阈值必须在0到1之间
            }
            DetectionThreshold = threshold;
        }
    }
}
