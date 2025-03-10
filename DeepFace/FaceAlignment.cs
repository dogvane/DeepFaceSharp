using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;
using DeepFace.Models;
using System.Diagnostics;

namespace DeepFace
{
    /// <summary>
    /// 人脸对齐工具类
    /// </summary>
    public class FaceAlignment
    {
        /// <summary>
        /// 提取并对齐人脸
        /// </summary>
        /// <param name="facialArea">人脸区域</param>
        /// <param name="img">原始图像</param>
        /// <param name="align">是否对齐</param>
        /// <param name="expandPercentage">扩展百分比</param>
        /// <returns>对齐后的人脸Mat</returns>
        public static Mat ExtractFace(
            FacialArea facialArea,
            Mat img,
            bool align = true,
            int expandPercentage = 0)
        {
            int x = facialArea.X;
            int y = facialArea.Y;
            int w = facialArea.W;
            int h = facialArea.H;
            
            // 从FacialArea中获取眼睛坐标并转换为Point类型
            Point? leftEye = facialArea.LeftEye.HasValue ? 
                new Point(facialArea.LeftEye.Value.x, facialArea.LeftEye.Value.y) : null;
            Point? rightEye = facialArea.RightEye.HasValue ? 
                new Point(facialArea.RightEye.Value.x, facialArea.RightEye.Value.y) : null;

            // 如果需要扩展人脸区域
            if (expandPercentage > 0)
            {
                // 按指定百分比扩展人脸区域的宽度和高度
                int expandedW = w + (int)(w * expandPercentage / 100.0);
                int expandedH = h + (int)(h * expandPercentage / 100.0);

                // 确保扩展区域不超出原图像边界
                x = Math.Max(0, x - (expandedW - w) / 2);
                y = Math.Max(0, y - (expandedH - h) / 2);
                w = Math.Min(img.Cols - x, expandedW);
                h = Math.Min(img.Rows - y, expandedH);
            }

            // 提取未对齐的检测人脸
            Mat detectedFace = new Mat(img, new Rect(x, y, w, h));

            // 如果不需要对齐，直接返回
            if (!align || leftEye == null || rightEye == null)
            {
                return detectedFace;
            }

            // 对齐处理
            // 提取带边界的子图像
            var (subImg, relativeX, relativeY) = ExtractSubImage(img, (x, y, w, h));
            
            // 根据眼睛坐标对齐图像
            var (alignedSubImg, angle) = AlignImageWithEyes(
                subImg, 
                leftEye.Value, 
                rightEye.Value);
                
            // 人脸对齐角度调整日志
            // Face alignment adjustment angle
            Debug.WriteLine($"Face alignment adjustment angle: {angle:F2}°");
            // 人脸对齐角度调整日志
            // Face alignment adjustment angle
            Console.WriteLine($"Face alignment adjustment angle: {angle:F2}°");

            // 投影旋转后的人脸区域坐标
            var (rotatedX1, rotatedY1, rotatedX2, rotatedY2) = ProjectFacialArea(
                (relativeX, relativeY, relativeX + w, relativeY + h),
                angle,
                (subImg.Rows, subImg.Cols));

            // 提取旋转对齐后的人脸区域
            Mat alignedFace = new Mat(alignedSubImg, 
                new Rect(rotatedX1, rotatedY1, rotatedX2 - rotatedX1, rotatedY2 - rotatedY1));

            // 释放临时变量
            subImg.Dispose();
            alignedSubImg.Dispose();
            
            return alignedFace;
        }

        /// <summary>
        /// 获取包含给定人脸区域的子图像，同时扩展人脸区域以确保对齐不会将人脸移出图像范围
        /// </summary>
        private static (Mat image, int relativeX, int relativeY) ExtractSubImage(
            Mat img, 
            (int x, int y, int w, int h) facialArea)
        {
            int x = facialArea.x;
            int y = facialArea.y;
            int w = facialArea.w;
            int h = facialArea.h;
            
            // 计算相对位置，人脸区域的宽度和高度的一半
            int relativeX = (int)(0.5 * w);
            int relativeY = (int)(0.5 * h);

            // 计算扩展坐标
            int x1 = x - relativeX;
            int y1 = y - relativeY;
            int x2 = x + w + relativeX;
            int y2 = y + h + relativeY;

            // 如果扩展区域适合图像内部
            if (x1 >= 0 && y1 >= 0 && x2 <= img.Cols && y2 <= img.Rows)
            {
                return (new Mat(img, new Rect(x1, y1, x2 - x1, y2 - y1)), relativeX, relativeY);
            }

            // 需要添加黑色像素填充的情况
            // 确保坐标在边界内
            x1 = Math.Max(0, x1);
            y1 = Math.Max(0, y1);
            x2 = Math.Min(img.Cols, x2);
            y2 = Math.Min(img.Rows, y2);
            
            Mat croppedRegion = new Mat(img, new Rect(x1, y1, x2 - x1, y2 - y1));
            
            // 创建一个黑色图像
            Mat extractedFace = new Mat(
                h + 2 * relativeY, 
                w + 2 * relativeX, 
                img.Type(), 
                new Scalar(0, 0, 0));

            // 映射裁剪区域
            int startX = Math.Max(0, relativeX - x);
            int startY = Math.Max(0, relativeY - y);
            
            // 创建目标ROI
            using var roi = new Mat(extractedFace, 
                new Rect(startX, startY, croppedRegion.Cols, croppedRegion.Rows));
            
            // 复制裁剪区域到目标ROI
            croppedRegion.CopyTo(roi);
            
            return (extractedFace, relativeX, relativeY);
        }

        /// <summary>
        /// 根据左右眼位置水平对齐给定图像
        /// </summary>
        private static (Mat image, float angle) AlignImageWithEyes(
            Mat img, 
            Point leftEye, 
            Point rightEye)
        {
            // 如果图像尺寸为零，则返回原始图像
            if (img.Rows == 0 || img.Cols == 0)
            {
                return (img.Clone(), 0);
            }

            // 计算旋转角度 - 修正计算方式，确保正确的眼睛对齐方向
            // 正确的计算应该是右眼到左眼的方向
            float angle = (float)Math.Atan2(rightEye.Y - leftEye.Y, rightEye.X - leftEye.X) * 180.0f / (float)Math.PI;

            // 获取图像中心
            int h = img.Rows;
            int w = img.Cols;
            Point2f center = new Point2f(w / 2.0f, h / 2.0f);

            // 获取旋转矩阵
            Mat rotMatrix = Cv2.GetRotationMatrix2D(center, angle, 1.0);
            
            // 创建结果图像
            Mat result = new Mat();
            
            // 应用仿射变换
            Cv2.WarpAffine(
                img, 
                result, 
                rotMatrix, 
                new Size(w, h), 
                flags: InterpolationFlags.Cubic, 
                borderMode: BorderTypes.Constant, 
                borderValue: new Scalar(0, 0, 0));

            // 释放旋转矩阵
            rotMatrix.Dispose();

            return (result, angle);
        }

        /// <summary>
        /// 在图像根据眼睛位置旋转后更新预计算的人脸区域坐标
        /// </summary>
        private static (int x1, int y1, int x2, int y2) ProjectFacialArea(
            (int x1, int y1, int x2, int y2) facialArea,
            float angle,
            (int height, int width) size)
        {
            // 规范化角度
            int direction = angle >= 0 ? 1 : -1;
            angle = Math.Abs(angle) % 360;
            
            if (angle == 0)
            {
                return facialArea;
            }

            // 角度转为弧度
            double angleRad = angle * Math.PI / 180.0;

            int height = size.height;
            int width = size.width;

            // 将人脸区域平移到图像中心
            double x = (facialArea.x1 + facialArea.x2) / 2.0 - width / 2.0;
            double y = (facialArea.y1 + facialArea.y2) / 2.0 - height / 2.0;

            // 旋转人脸区域
            double xNew = x * Math.Cos(angleRad) + y * direction * Math.Sin(angleRad);
            double yNew = -x * direction * Math.Sin(angleRad) + y * Math.Cos(angleRad);

            // 将人脸区域平移回原始位置
            xNew = xNew + width / 2.0;
            yNew = yNew + height / 2.0;

            // 计算对齐后的人脸区域投影坐标
            int x1 = (int)(xNew - (facialArea.x2 - facialArea.x1) / 2.0);
            int y1 = (int)(yNew - (facialArea.y2 - facialArea.y1) / 2.0);
            int x2 = (int)(xNew + (facialArea.x2 - facialArea.x1) / 2.0);
            int y2 = (int)(yNew + (facialArea.y2 - facialArea.y1) / 2.0);

            // 验证投影坐标在图像边界内
            x1 = Math.Max(x1, 0);
            y1 = Math.Max(y1, 0);
            x2 = Math.Min(x2, width);
            y2 = Math.Min(y2, height);

            return (x1, y1, x2, y2);
        }
    }
}
