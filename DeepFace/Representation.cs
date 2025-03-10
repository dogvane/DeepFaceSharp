using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;
using DeepFace.Factory;
using DeepFace.Models;

namespace DeepFace
{
    /// <summary>
    /// 
    /// </summary>
    public class RepresentationResult
    {
        /// <summary>
        /// 人脸嵌入向量
        /// </summary>
        public float[] Embedding { get; set; }

        /// <summary>
        /// 人脸检测结果
        /// </summary>
        public DetectionResult DetectionResult { get; set; }
    }

    /// <summary>
    /// 人脸检测类，用于提取人脸特征
    /// </summary>
    public partial class DeepFace
    {
        /// <summary>
        /// 提取图像中人脸的特征表示
        /// </summary>
        /// <param name="imgPath">图像文件路径</param>
        /// <param name="maxFaces">最大处理的人脸数量，null表示处理所有人脸</param>
        /// <returns></returns>
        public List<RepresentationResult> Represent(string imgPath, int? maxFaces = null)
        {
            if (!File.Exists(imgPath))
            {
                throw new FileNotFoundException("Image file does not exist", imgPath);
            }

            // 人脸检测
            var detectionResults = ExtractFaces(imgPath);

            if (maxFaces.HasValue)
            {
                detectionResults = detectionResults
                    .OrderByDescending(x => x.Confidence)
                    .Take(maxFaces.Value).ToList();
            }

            var result = new List<RepresentationResult>();

            foreach (var detectionResult in detectionResults)
            {
                RepresentationResult representationResult = new RepresentationResult();
                representationResult.DetectionResult = detectionResult;
                representationResult.Embedding = _recognizer.GetEmbedding(detectionResult.Face);
                result.Add(representationResult);
            }

            return result;
        }
    }
}
