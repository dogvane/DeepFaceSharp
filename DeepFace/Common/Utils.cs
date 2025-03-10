using DeepFace.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepFace.Common
{
    public static class Utils
    {

        static private float CalculateIOU(DetectionResult box1, DetectionResult box2)
        {
            float x1 = Math.Max(box1.FacialArea.X, box2.FacialArea.X);
            float y1 = Math.Max(box1.FacialArea.Y, box2.FacialArea.Y);
            float x2 = Math.Min(box1.FacialArea.X + box1.FacialArea.W, box2.FacialArea.X + box2.FacialArea.W);
            float y2 = Math.Min(box1.FacialArea.Y + box1.FacialArea.H, box2.FacialArea.Y + box2.FacialArea.H);

            float intersectionArea = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
            float box1Area = box1.FacialArea.W * box1.FacialArea.H;
            float box2Area = box2.FacialArea.W * box2.FacialArea.H;
            float unionArea = box1Area + box2Area - intersectionArea;

            return intersectionArea / unionArea;
        }

        public static List<DetectionResult> ApplyNMS(this List<DetectionResult> boxes, float iouThreshold = 0.3f)
        {
            var sortedBoxes = boxes.OrderByDescending(x => x.Confidence).ToList();
            var selected = new List<DetectionResult>();

            while (sortedBoxes.Count > 0)
            {
                var current = sortedBoxes[0];
                selected.Add(current);
                sortedBoxes.RemoveAt(0);

                sortedBoxes = sortedBoxes.Where(box =>
                    CalculateIOU(current, box) < iouThreshold
                ).ToList();
            }

            return selected;
        }

    }
}
