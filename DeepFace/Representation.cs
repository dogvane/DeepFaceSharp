using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenCvSharp;
using DeepFace.Factory;
using DeepFace.Models;

namespace DeepFace
{
    public class Representation
    {
        public class FacialArea
        {
            public int X { get; set; }
            public int Y { get; set; }
            public int Width { get; set; }
            public int Height { get; set; }
        }

        public class RepresentationResult
        {
            public float[] Embedding { get; set; }
            public FacialArea FacialArea { get; set; }
            public float FaceConfidence { get; set; }
        }

        public RepresentationResult Represent(
            string imgPath,
            string modelName = "arcface",
            bool enforceDetection = true,
            string detectorBackend = "yunet",
            bool align = true,
            int expandPercentage = 0,
            string normalization = "base",
            bool antiSpoofing = false,
            int? maxFaces = null)
        {
            if (!File.Exists(imgPath))
            {
                throw new FileNotFoundException("图像文件不存在", imgPath);
            }

            // 使用工厂方法创建检测器和识别器
            var detector = FaceFactory.CreateDetector(detectorBackend);
            var recognizer = FaceFactory.CreateRecognizer(modelName);

            // 人脸检测
            var detectionResults = detector.DetectFaces(imgPath);
            
            if (!detectionResults.Any())
            {
                if (enforceDetection)
                {
                    throw new Exception("未检测到人脸");
                }
                return null;
            }

            // 获取最大置信度的人脸
            var bestFace = detectionResults.OrderByDescending(x => x.Confidence).First();
            
            // 获取人脸特征向量
            var embedding = recognizer.GetEmbedding(imgPath);

            return new RepresentationResult
            {
                Embedding = embedding,
                FacialArea = new FacialArea
                {
                    X = bestFace.FacialArea.X,
                    Y = bestFace.FacialArea.Y,
                    Width = bestFace.FacialArea.W,
                    Height = bestFace.FacialArea.H
                },
                FaceConfidence = bestFace.Confidence
            };
        }
    }
}
