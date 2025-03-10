using System.Drawing;
using System.Text.Json;
using System.Text;
using System;
using DeepFace.Config;
using DeepFace.Models;
using OpenCvSharp;
using Point = OpenCvSharp.Point;
using DeepFace.Models.FacialRecognition;
using DeepFace.Common;

namespace DeepFace
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //TestYuNet();
            //yolov8();
            // yolov11();
            DetectionTest();
            // TestArcFace();
            //FacenetTest();
        }

        /// <summary>
        /// 绘制人脸检测框和特征点
        /// </summary>
        /// <param name="image">图像</param>
        /// <param name="face">检测到的人脸</param>
        private static void DrawFace(Mat image, DetectionResult face)
        {
            var area = face.FacialArea;
            // 绘制人脸检测框
            Cv2.Rectangle(image,
                new Point(area.X, area.Y),
                new Point(area.X + area.W, area.Y + area.H),
                Scalar.Red, 2);

            // 绘制特征点
            if (area.LeftEye.HasValue)
                Cv2.Circle(image, new Point(area.LeftEye.Value.x, area.LeftEye.Value.y), 2, Scalar.Green, -1);
            if (area.RightEye.HasValue)
                Cv2.Circle(image, new Point(area.RightEye.Value.x, area.RightEye.Value.y), 2, Scalar.Green, -1);
            if (area.Nose.HasValue)
                Cv2.Circle(image, new Point(area.Nose.Value.x, area.Nose.Value.y), 2, Scalar.Green, -1);
            if (area.MouthLeft.HasValue)
                Cv2.Circle(image, new Point(area.MouthLeft.Value.x, area.MouthLeft.Value.y), 2, Scalar.Green, -1);
            if (area.MouthRight.HasValue)
                Cv2.Circle(image, new Point(area.MouthRight.Value.x, area.MouthRight.Value.y), 2, Scalar.Green, -1);

            Console.WriteLine(JsonSerializer.Serialize(face));
        }

        private static void yolov11()
        {
            ModelDownloadUtils.ProxyServer = "http://192.168.1.3:10811";
            ModelConfiguration.Instance.SetDetectionThreshold(0.6f);
            var yunet = new Yolo(Yolo.YoloModel.YoloV11NFace);

            var fileName = "../../../../SampleImages/000000001000.jpg";
            var image = Cv2.ImRead(fileName);
            var ret = yunet.DetectFaces(image);
            Console.WriteLine(ret.Count);
            foreach (var face in ret)
            {
                DrawFace(image, face);
            }

            // 保存标记后的图片
            Cv2.ImWrite(new FileInfo(fileName).Name.Replace(".jpg", "_yolo11.jpg"), image);
        }

        private static void yolov8()
        {
            ModelConfiguration.Instance.SetDetectionThreshold(0.6f);
            var yunet = new Yolo();
            var fileName = "../../../../SampleImages/000000001000.jpg";

            var image = Cv2.ImRead(fileName);
            var ret = yunet.DetectFaces(image);
            Console.WriteLine(ret.Count);
            foreach (var face in ret)
            {
                DrawFace(image, face);
            }

            // 保存标记后的图片
            Cv2.ImWrite(new FileInfo(fileName).Name.Replace(".jpg", "_yolo8.jpg"), image);
        }

        private static void DetectionTest()
        {
            ModelConfiguration.Instance.SetDetectionThreshold(0.8f);
            var deepface = new DeepFace();
            var sourceimg = "../../../../SampleImages/000000001000.jpg";
            bool align = true;
            var ret = deepface.ExtractFaces(sourceimg,
                align: align,
                expandPercentage: 20);

            var index = 0;
            foreach (var face in ret)
            {
                Console.WriteLine(JsonSerializer.Serialize(face));
                face.Face.SaveImage(sourceimg.Replace(".jpg", $"_align_{align}_{index++}.jpg"));
            }

            align = false;
            var ret2 = deepface.ExtractFaces(sourceimg,
                align: align,
                expandPercentage: 20);

            var index2 = 0;
            foreach (var face in ret2)
            {
                Console.WriteLine(JsonSerializer.Serialize(face));
                face.Face.SaveImage(sourceimg.Replace(".jpg", $"_align_{align}_{index2++}.jpg"));
            }
        }

        private static void FacenetTest()
        {
            var facenet = new Facenet();
            var fileName = "../../../../SampleImages/000000001000_align_False_4.jpg";

            var image = Cv2.ImRead(fileName);
            var ret = facenet.GetEmbedding(image);
            Console.WriteLine(ret.Length);
            var txt = string.Join("\n", ret);
            File.WriteAllText("facenet128_net_onnx.txt", txt);
        }

        private static void TestArcFace()
        {
            var arcFace = new ArcFace();
            var fileName = "../../../../SampleImages/000000001000_align_False_4.jpg";

            var image = Cv2.ImRead(fileName);
            var ret = arcFace.GetEmbedding(image);
            Console.WriteLine(ret.Length);
            var txt = string.Join("\n", ret);
            File.WriteAllText("arcface_net_onnx.txt", txt);
        }

        private static void TestYuNet()
        {
            var yunet = new YuNet();
            var fileName = "../../../../SampleImages/000000001000.jpg";

            var image = Cv2.ImRead(fileName);
            var ret = yunet.DetectFaces(image);
            Console.WriteLine("find face: " + ret.Count);
            foreach (var face in ret)
            {
                DrawFace(image, face);
            }

            // 保存标记后的图片
            Cv2.ImWrite(new FileInfo(fileName).Name.Replace(".jpg", "_YuNet.jpg"), image);
        }

    }
}
