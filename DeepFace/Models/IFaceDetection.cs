using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenCvSharp;

namespace DeepFace.Models
{
    public interface IDetection
    {
        List<DetectionResult> DetectFaces(string imagePath);
        List<DetectionResult> DetectFaces(Mat image);
    }
}