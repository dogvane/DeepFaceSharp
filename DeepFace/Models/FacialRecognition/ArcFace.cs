using DeepFace.Common;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepFace.Models.FacialRecognition
{
    public class ArcFace:IDisposable,IRecognition
    {
        private readonly InferenceSession _session;
        static string ModelUrl { get; set; } = "https://github.com/dogvane/DeepFaceSharp/releases/download/v0.0.0/arcface.onnx";

        public class Config
        {
            public string ModelPath { get; set; } = "models/arcface.onnx";
            public Config() { }
        }

        private Config _config;

        public ArcFace(Config config = null)
        {
            if (config == null)
            {
                _config = new Config();
            }
            else
            {
                _config = config;
            }
            
            if (File.Exists(_config.ModelPath))
            {
                _session = new InferenceSession(_config.ModelPath);
                _session.PrintOnnxMetadata();
            }
            else
            {
                // 尝试从网络下载模型
                Console.WriteLine($"Model file not found at {_config.ModelPath}, attempting to download from {ModelUrl}");
                try
                {
                    string downloadedPath = ModelDownloadUtils.EnsureModelExists(_config.ModelPath, ModelUrl);
                    Console.WriteLine($"Model downloaded successfully to {downloadedPath}");
                    
                    _session = new InferenceSession(downloadedPath);
                    _session.PrintOnnxMetadata();
                }
                catch (Exception ex)
                {
                    throw new FileNotFoundException($"Model file not found and download failed. {_config.ModelPath}. Error: {ex.Message}", ex);
                }
            }
        }

        public float[] GetEmbeddingByMat(Mat image)
        {
            var resizedImage = image.Resize(112, 112).Image;
            // 创建输入张量
            var inputTensor = ImageUtils.ImgToTensortTF(resizedImage);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            // 运行推理
            using var output = _session.Run(inputs);
            
            // 获取输出嵌入向量
            var embedding = output.First().AsEnumerable<float>().ToArray();

            return embedding;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }

        public float[] GetEmbedding(string imagePath)
        {
            var mat = Cv2.ImRead(imagePath);
            return GetEmbeddingByMat(mat);
        }
    }
}
