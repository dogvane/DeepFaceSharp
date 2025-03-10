using OpenCvSharp;
using System.Text.Json.Serialization;

namespace DeepFace.Models
{
    /// <summary>
    /// 人脸检测结果
    /// </summary>
    public class DetectionResult
    {
        /// <summary>
        /// 检测到的人脸图像
        /// </summary>
        [JsonIgnore]
        public Mat Face { get; set; }

        /// <summary>
        /// 人脸区域信息，包含位置和关键点
        /// </summary>
        public FacialArea FacialArea { get; set; }

        /// <summary>
        /// 人脸检测的置信度分数 (0-1之间)
        /// </summary>
        public float Confidence { get; set; }

        /// <summary>
        /// 获取人脸区域的Mat对象
        /// </summary>
        /// <param name="originalImage">原始图像</param>
        /// <returns>人脸区域的Mat对象，如果已有Face属性则直接返回</returns>
        public Mat GetFaceMat(Mat originalImage)
        {
            if (Face != null)
            {
                return Face.Clone();
            }

            var faceRect = new Rect(FacialArea.X, FacialArea.Y, FacialArea.W, FacialArea.H);
            return new Mat(originalImage, faceRect);
        }
    }
}
