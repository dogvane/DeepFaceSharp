using DeepFace.Common;
using DeepFace.Config;
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

        public class Config : ModelBaseConfig
        {
            /// <summary>
            /// 目前的已有的模型只支持128和512维的嵌入向量
            /// 模型文件需要自己指定到对应不同的维度
            /// 但输入的图片尺寸仍然维持 160x160
            /// </summary>
            public EmbeddingDimension EmbeddingSize { get; set; }

            public Config(EmbeddingDimension embedding = EmbeddingDimension.Dimension128)
            {
                EmbeddingSize = embedding;
                ModelFile = ModelConfiguration.Instance.ModelsDirectory;

                switch (embedding)
                {
                    case EmbeddingDimension.Dimension128:
                        ModelFile = Path.Combine(ModelFile, "facenet128_weights.onnx");
                        ModelUrl = "https://github.com/dogvane/DeepFaceSharp/releases/download/v0.0.0/facenet128_weights.onnx";
                        break;
                    case EmbeddingDimension.Dimension512:
                        ModelFile = Path.Combine(ModelFile, "facenet512_weights.onnx");
                        ModelUrl = "https://github.com/dogvane/DeepFaceSharp/releases/download/v0.0.0/facenet512_weights.onnx";
                        break;
                    default:
                        throw new ArgumentException("Unsupported embedding size");
                }
            }
        }

        private Config _config;

        public Facenet() : this(new Config())
        {

        }

        public Facenet(Config config)
        {
            _config = config;

            // 检查模型文件是否存在，不存在则尝试下载
            ModelDownloadUtils.EnsureModelExists(_config.ModelFile, _config.ModelUrl);
            _session = OnnxUtils.CreateSession(_config.ModelFile, _config.DeviceId, _config.PreferredBackend);
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