using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DeepFace.Common
{
    public class ResizeResult
    {
        public Mat Image { get; set; }
        public bool IsResized { get; set; }
        public float Ratio { get; set; }
        public int OffsetX { get; set; }
        public int OffsetY { get; set; }
    }

    public static class ImageUtils
    {
        public static ResizeResult Resize(this Mat image, int width, int height)
        {
            if (image.Height != height || image.Width != width)
            {
                var r = Math.Min((float)height / image.Height, (float)width / image.Width);
                var newSize = new Size(image.Width * r, image.Height * r);
                var resizedImage = image.Resize(newSize);

                var processedImage = new Mat(new Size(width, height), image.Type(), Scalar.Black);
                var offsetX = (width - resizedImage.Width) / 2;
                var offsetY = (height - resizedImage.Height) / 2;
                var roi = new Rect(offsetX, offsetY, resizedImage.Width, resizedImage.Height);
                resizedImage.CopyTo(new Mat(processedImage, roi));

                return new ResizeResult
                {
                    Image = processedImage,
                    IsResized = true,
                    Ratio = r,
                    OffsetX = offsetX,
                    OffsetY = offsetY
                };
            }
            else
            {
                return new ResizeResult
                {
                    Image = image,
                    IsResized = false,
                    Ratio = 1f,
                    OffsetX = 0,
                    OffsetY = 0
                };
            }
        }

        /// <summary>
        /// 将图像转换为Channel First格式的张量 (NCHW格式: [batch, channels, height, width])
        /// 适用于PyTorch等使用NCHW格式的模型
        /// </summary>
        public static DenseTensor<float> ImgToTensortPY(Mat outImg, bool useStd = false, bool normalize = true)
        {
            // 检查并转换图像格式
            Mat processedImg = outImg;

            int height = processedImg.Height;
            int width = processedImg.Width;
            int channels = processedImg.Channels();
            int totalSize = height * width * channels;

            // 创建临时数组并复制数据
            var paddedArray = new byte[totalSize];
            Marshal.Copy(processedImg.Data, paddedArray, 0, totalSize);

            // 创建结果数组
            var chwArray = new float[channels * height * width];
            var paddedSpan = paddedArray.AsSpan();
            var resultSpan = chwArray.AsSpan();

            // 计算每个通道的大小
            int channelSize = height * width;

            // 对每个像素进行处理
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int srcIdx = (h * width + w) * channels;
                    for (int c = 0; c < channels; c++)
                    {
                        int dstIdx = c * channelSize + h * width + w;
                        if (useStd)
                        {
                            float pixel = paddedSpan[srcIdx + c];
                            float stdValue = c switch
                            {
                                0 => (pixel - 123.676f) / 58.395f,
                                1 => (pixel - 116.28f) / 57.12f,
                                2 => (pixel - 103.53f) / 57.375f,
                                _ => pixel
                            };
                            resultSpan[dstIdx] = normalize ? stdValue / 255f : stdValue; 
                        }
                        else
                        {
                            resultSpan[dstIdx] = normalize ? paddedSpan[srcIdx + c] / 255f : paddedSpan[srcIdx + c]; 
                        }
                    }
                }
            }

            return new DenseTensor<float>(chwArray, [1, channels, height, width]);
        }


        /// <summary>
        /// 将图像转换为Channel Last格式的张量 (NHWC格式: [batch, height, width, channels])
        /// 适用于TensorFlow等使用NHWC格式的模型
        /// </summary>
        public static DenseTensor<float> ImgToTensortTF(Mat outImg, bool useStd = false, bool normalize = true)
        {
            // 检查并转换图像格式
            Mat processedImg = outImg;

            int height = processedImg.Height;
            int width = processedImg.Width;
            int channels = processedImg.Channels();
            int totalSize = height * width * channels;

            // 创建临时数组并复制数据
            var paddedArray = new byte[totalSize];
            Marshal.Copy(processedImg.Data, paddedArray, 0, totalSize);

            // 创建结果数组 - TensorFlow格式为NHWC
            var hwcArray = new float[height * width * channels];
            var paddedSpan = paddedArray.AsSpan();
            var resultSpan = hwcArray.AsSpan();

            // 对每个像素进行处理
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int srcIdx = (h * width + w) * channels;
                    int dstIdx = (h * width + w) * channels;
                    
                    for (int c = 0; c < channels; c++)
                    {
                        if (useStd)
                        {
                            float pixel = paddedSpan[srcIdx + c];
                            float stdValue = c switch
                            {
                                0 => (pixel - 123.676f) / 58.395f,
                                1 => (pixel - 116.28f) / 57.12f,
                                2 => (pixel - 103.53f) / 57.375f,
                                _ => pixel
                            };
                            resultSpan[dstIdx + c] = normalize ? stdValue / 255f : stdValue;
                        }
                        else
                        {
                            resultSpan[dstIdx + c] = normalize ? paddedSpan[srcIdx + c] / 255f : paddedSpan[srcIdx + c];
                        }
                    }
                }
            }

            return new DenseTensor<float>(hwcArray, new[] { 1, height, width, channels });
        }

        public static unsafe DenseTensor<float> ImgToTensort2(Mat outImg, bool useStd = false)
        {
            byte[] paddedArray = new byte[outImg.Total() * outImg.Channels()];

            Marshal.Copy(outImg.Data, paddedArray, 0, paddedArray.Length);

            // todo 这里需要做一下性能测试

            float[] chwArray = new float[3 * outImg.Height * outImg.Width];
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < outImg.Height; h++)
                {
                    for (int w = 0; w < outImg.Width; w++)
                    {
                        if (useStd)
                        {
                            switch (c)
                            {
                                case 0:
                                    chwArray[c * outImg.Height * outImg.Width + h * outImg.Width + w] = (paddedArray[h * outImg.Width * 3 + w * 3 + c] - 123.676f) / 58.395f;
                                    break;
                                case 1:
                                    chwArray[c * outImg.Height * outImg.Width + h * outImg.Width + w] = (paddedArray[h * outImg.Width * 3 + w * 3 + c] - 116.28f) / 57.12f;
                                    break;
                                case 2:
                                    chwArray[c * outImg.Height * outImg.Width + h * outImg.Width + w] = (paddedArray[h * outImg.Width * 3 + w * 3 + c] - 103.53f) / 57.375f;
                                    break;
                            }

                        }
                        else
                            chwArray[c * outImg.Height * outImg.Width + h * outImg.Width + w] = paddedArray[h * outImg.Width * 3 + w * 3 + c];
                    }
                }
            }

            var imageShpae = new int[] { 1, 3, outImg.Height, outImg.Width };
            return new DenseTensor<float>(chwArray, imageShpae);
        }

        /// <summary>
        /// 有问题的代码
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        public static DenseTensor<float> ConvertMatToTensor(Mat image)
        {
            // 转换为RGB格式
            var rgb = new Mat();
            //Cv2.CvtColor(image, rgb, ColorConversionCodes.BGR2RGB);

            // 创建tensor
            var dimensions = new[] { 1, 3, image.Height, image.Width };
            var tensor = new DenseTensor<float>(dimensions);
            var span = tensor.Buffer.Span;

            int pixelCount = image.Height * image.Width;
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = image.At<Vec3b>(y, x);
                    int idx = y * image.Width + x;
                    span[idx] = pixel[0] / 255.0f;                           // R
                    span[pixelCount + idx] = pixel[1] / 255.0f;             // G
                    span[2 * pixelCount + idx] = pixel[2] / 255.0f;         // B
                }
            }

            return tensor;
        }

        public static DenseTensor<float> ConvertMatToTensorTF(Mat image)
        {
            // 创建tensor - TensorFlow格式为NHWC
            var dimensions = new[] { 1, image.Height, image.Width, 3 };
            var tensor = new DenseTensor<float>(dimensions);
            var span = tensor.Buffer.Span;

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = image.At<Vec3b>(y, x);
                    int idx = (y * image.Width + x) * 3;  // 每个像素连续存储3个通道
                    span[idx] = pixel[0] / 255.0f;     // R
                    span[idx + 1] = pixel[1] / 255.0f; // G
                    span[idx + 2] = pixel[2] / 255.0f; // B
                }
            }

            return tensor;
        }
    }
}
