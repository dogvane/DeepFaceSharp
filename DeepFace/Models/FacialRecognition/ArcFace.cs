using DeepFace.Common;
using DeepFace.Config;
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
    public class ArcFace : IDisposable, IRecognition
    {
        private readonly InferenceSession _session;

        public class Config : ModelBaseConfig
        {
            public Config()
            {
                var path = ModelConfiguration.Instance.ModelsDirectory;
                ModelFile = Path.Combine(path, "arcface.onnx");
                ModelUrl = "https://github.com/dogvane/DeepFaceSharp/releases/download/v0.0.0/arcface.onnx";
            }
        }

        private Config _config;

        public ArcFace() : this(new Config())
        {

        }

        public ArcFace(Config config)
        {
            _config = config;

            // 检查模型文件是否存在，不存在则尝试下载
            ModelDownloadUtils.EnsureModelExists(_config.ModelFile, _config.ModelUrl);
            _session = OnnxUtils.CreateSession(_config.ModelFile, _config.DeviceId, _config.PreferredBackend);
        }

        public float[] GetEmbedding(Mat image)
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
            using var mat = Cv2.ImRead(imagePath);
            return GetEmbedding(mat);
        }
    }
}
