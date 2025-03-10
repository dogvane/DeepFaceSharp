using DeepFace.Common;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace DeepFace.Models.FacialRecognition
{
    public class Facenet : IDisposable, IRecognition
    {
        private readonly InferenceSession _session;

        /// <summary>
        /// 嵌入向量维度的枚举
        /// </summary>
        public enum EmbeddingDimension
        {
            /// <summary>
            /// 128维嵌入向量
            /// </summary>
            Dimension128 = 128,
            
            /// <summary>
            /// 512维嵌入向量
            /// </summary>
            Dimension512 = 512
        }

        public class Config
        {
            public string ModelPath { get; set; }

            /// <summary>
            /// 目前的已有的模型只支持128和512维的嵌入向量
            /// 模型文件需要自己指定到对应不同的维度
            /// 但输入的图片尺寸仍然维持 160x160
            /// </summary>
            public EmbeddingDimension EmbeddingSize { get; set; }
            
            /// <summary>
            /// 模型不存在时是否自动从网络下载
            /// </summary>
            public bool AutoDownload { get; set; } = false;
            
            /// <summary>
            /// 模型文件的预期MD5值，用于校验下载的模型是否正确
            /// </summary>
            public string ExpectedMd5 { get; set; } = null;

            public Config(EmbeddingDimension embedding = EmbeddingDimension.Dimension128)
            {
                EmbeddingSize = embedding;
                switch(embedding)
                {
                    case EmbeddingDimension.Dimension128:
                        ModelPath = "models/facenet128_weights.onnx";
                        break;
                    case EmbeddingDimension.Dimension512:
                        ModelPath = "models/facenet512_weights.onnx";
                        break;
                    default:
                        throw new ArgumentException("Unsupported embedding size");
                }
            }
        }

        private Config _config;
        
        // 模型下载链接
        private static readonly Dictionary<EmbeddingDimension, string> ModelUrls = new Dictionary<EmbeddingDimension, string>
        {
            { EmbeddingDimension.Dimension128, "https://github.com/dogvane/DeepFaceSharp/releases/download/v0.0.0/facenet128_weights.onnx" },
            { EmbeddingDimension.Dimension512, "https://github.com/dogvane/DeepFaceSharp/releases/download/v0.0.0/facenet512_weights.onnx" }
        };

        public Facenet(Config config = null)
        {
            if (config == null)
            {
                _config = new Config() { AutoDownload = true };
            }
            else
            {
                _config = config;
            }
            
            // 检查模型文件是否存在，不存在则尝试下载
            if (!File.Exists(_config.ModelPath) && _config.AutoDownload)
            {
                if (ModelUrls.TryGetValue(_config.EmbeddingSize, out string downloadUrl))
                {
                    Console.WriteLine($"Model file not found. Attempting to download {(int)_config.EmbeddingSize}d model...");
                    try
                    {
                        ModelDownloadUtils.EnsureModelExists(
                            _config.ModelPath, 
                            downloadUrl, 
                            expectedMd5: _config.ExpectedMd5
                        );
                    }
                    catch (Exception ex)
                    {
                        throw new Exception($"Failed to download model: {ex.Message}", ex);
                    }
                }
                else
                {
                    throw new Exception($"No download URL defined for embedding size {(int)_config.EmbeddingSize}");
                }
            }
            
            if (File.Exists(_config.ModelPath))
            {
                _session = new InferenceSession(_config.ModelPath);
                var outSize = _session.OutputMetadata.First().Value.Dimensions[1];
                if (outSize != (int)_config.EmbeddingSize)
                {
                    _session.PrintOnnxMetadata();
                    throw new Exception($"The model does not support the specified embedding size. onnx.out:{outSize} config.use:{(int)_config.EmbeddingSize}");
                }
            }
            else
            {
                throw new FileNotFoundException("Model file not found. " + _config.ModelPath);
            }
        }
        public float[] GetEmbedding(Mat image)
        {
            var resizedImage = image.Resize(160, 160).Image;

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
            using var mat = Cv2.ImRead(imagePath);
            return GetEmbedding(mat);
        }
    }
}