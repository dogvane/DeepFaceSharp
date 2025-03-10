namespace DeepFace.Models
{
    /// <summary>
    /// 表示检测到的人脸区域信息
    /// </summary>
    public class FacialArea
    {
        /// <summary>
        /// 人脸区域左上角的X坐标
        /// </summary>
        public int X { get; set; }

        /// <summary>
        /// 人脸区域左上角的Y坐标
        /// </summary>
        public int Y { get; set; }

        /// <summary>
        /// 人脸区域的宽度
        /// </summary>
        public int W { get; set; }

        /// <summary>
        /// 人脸区域的高度
        /// </summary>
        public int H { get; set; }

        /// <summary>
        /// 左眼位置坐标 (相对于人脸本身而非观察者视角)
        /// </summary>
        public (int x, int y)? LeftEye { get; set; }

        /// <summary>
        /// 右眼位置坐标 (相对于人脸本身而非观察者视角)
        /// </summary>
        public (int x, int y)? RightEye { get; set; }

        /// <summary>
        /// 鼻子位置坐标
        /// </summary>
        public (int x, int y)? Nose { get; set; }

        /// <summary>
        /// 嘴巴左侧位置坐标
        /// </summary>
        public (int x, int y)? MouthLeft { get; set; }

        /// <summary>
        /// 嘴巴右侧位置坐标
        /// </summary>
        public (int x, int y)? MouthRight { get; set; }
    }
}
