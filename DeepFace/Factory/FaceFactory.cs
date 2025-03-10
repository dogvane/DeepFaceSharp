using DeepFace.Models;
using DeepFace.Models.FacialRecognition;
using DeepFace.Config;

namespace DeepFace.Factory
{
    public static class FaceFactory
    {
        public static IDetection CreateDetector(string backend = "yunet")
        {
            return backend.ToLower() switch
            {
                "yunet" => new YuNet(),
                "yolo8" => new Yolo(Yolo.YoloModel.YoloV8NFace),
                "yolo11" => new Yolo(Yolo.YoloModel.YoloV11NFace),
                "yolo11s" => new Yolo(Yolo.YoloModel.YoloV11SFace),
                "yolo11n" => new Yolo(Yolo.YoloModel.YoloV11NFace),
                "yolo11m" => new Yolo(Yolo.YoloModel.YoloV11MFace),
                _ => throw new ArgumentException($"Unsupported detection backend: {backend}")
            };
        }

        public static IRecognition CreateRecognizer(string model = "arcface")
        {
            return model.ToUpper() switch
            {
                "arcface" => new ArcFace(),
                "facenet" => new Facenet(),
                _ => throw new ArgumentException($"Unsupported recognition model: {model}")
            };
        }
    }
}
