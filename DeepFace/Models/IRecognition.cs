using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace DeepFace.Models
{
    public interface IRecognition
    {
        /// <summary>
        /// 仅对图像做人脸编码，并不会再进行人脸检测
        /// </summary>
        /// <param name="imagePath"></param>
        /// <returns></returns>
        float[] GetEmbedding(string imagePath);

        float[] GetEmbedding(Mat image);
    }
}