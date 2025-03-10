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

    public partial class DeepFace
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="imgPath"></param>
        /// <param name="outputFaceMat">结果里是否输出人脸的图片</param>
        /// <param name="align">是否人脸对齐(仅输出人脸时生效)</param>
        /// <param name="expandPercentage">抽取人脸的扩展范围 0-100 (仅输出人脸时生效)</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public List<DetectionResult> ExtractFaces(string imgPath, bool outputFaceMat = true, bool?align = true, int? expandPercentage = 0)
        {
            if (string.IsNullOrEmpty(imgPath) || !File.Exists(imgPath))
            {
                throw new ArgumentException($"Invalid image path: {imgPath}");
            }

            // 加载图像
            using var sourceImage = Cv2.ImRead(imgPath);
            if (sourceImage.Empty())
            {
                throw new ArgumentException($"Failed to load image: {imgPath}");
            }

            // 检测人脸
            var detectionResults = _detector.DetectFaces(sourceImage);

            if (!detectionResults.Any())
            {
                Console.WriteLine("No face detected!");
                return detectionResults;
            }

            if (outputFaceMat)
            {
                var _align = _config.Align;
                var _expandPercentage = _config.ExpandPercentage;

                if (align != null)
                {
                    _align = align.Value;
                }

                if (expandPercentage != null)
                {
                    _expandPercentage = expandPercentage.Value;
                }
                // 使用新的人脸对齐算法处理每个检测结果
                foreach (var result in detectionResults)
                {
                    // 使用FaceAlignment类提取并对齐人脸
                    result.Face = FaceAlignment.ExtractFace(
                        result.FacialArea,
                        sourceImage,
                        _align,
                        _expandPercentage
                    );
                }
            }

            return detectionResults;
        }
    }
}
