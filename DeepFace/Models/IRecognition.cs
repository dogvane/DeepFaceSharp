using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace DeepFace.Models
{
    public interface IRecognition
    {
        float[] GetEmbedding(string imagePath);
    }
}