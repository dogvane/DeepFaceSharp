using System;
using System.IO;
using DeepFace.Config;

namespace DeepFace
{
    /// <summary>
    /// DeepFace主类，用于初始化和配置模型
    /// </summary>
    public static class DeepFace
    {
        /// <summary>
        /// 初始化DeepFace库
        /// </summary>
        /// <param name="modelsDirectory">模型目录路径，如果为null则使用默认路径</param>
        public static void Initialize(string modelsDirectory = null)
        {
            if (string.IsNullOrEmpty(modelsDirectory))
            {
                // 默认使用应用程序目录下的models文件夹
                modelsDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models");
            }
            
            // 确保目录存在
            if (!Directory.Exists(modelsDirectory))
            {
                Directory.CreateDirectory(modelsDirectory);
            }

            // 初始化模型配置
            ModelConfiguration.Instance.Initialize(modelsDirectory);
            
            Console.WriteLine($"DeepFace 初始化完成，模型目录: {modelsDirectory}");
        }

        /// <summary>
        /// 设置人脸检测的默认置信度阈值
        /// </summary>
        /// <param name="threshold">阈值(0-1之间的浮点数)</param>
        public static void SetDetectionThreshold(float threshold)
        {
            if (!ModelConfiguration.Instance.IsInitialized)
            {
                throw new InvalidOperationException("请先调用 DeepFace.Initialize() 方法初始化模型配置");
            }
            
            ModelConfiguration.Instance.SetDetectionThreshold(threshold);
            Console.WriteLine($"人脸检测置信度阈值已设置为: {threshold}");
        }
    }
}
