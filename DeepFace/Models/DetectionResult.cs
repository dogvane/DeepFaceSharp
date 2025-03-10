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
    }
}
