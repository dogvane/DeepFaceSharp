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
        /// ����ͼ�����������룬�������ٽ����������
        /// </summary>
        /// <param name="imagePath"></param>
        /// <returns></returns>
        float[] GetEmbedding(string imagePath);

        float[] GetEmbedding(Mat image);
    }
}