using Microsoft.ML.OnnxRuntime;

namespace DeepFace.Common
{
    public static class OnnxUtils
    {
        public static void PrintOnnxMetadata(this InferenceSession session)
        {
            // 打印模型信息
            // 模型输入节点信息
            Console.WriteLine("Model Input Nodes Information:");
            PrintNodesInfo(session.InputMetadata);

            // 模型输出节点信息
            Console.WriteLine("\nModel Output Nodes Information:");
            PrintNodesInfo(session.OutputMetadata);
        }

        private static void PrintNodesInfo(IReadOnlyDictionary<string, NodeMetadata> nodes)
        {
            foreach (var node in nodes)
            {
                // 名称
                Console.WriteLine($"Name: {node.Key}");
                // 维度
                Console.WriteLine($"Dimensions: [{string.Join(",", node.Value.Dimensions)}]");
                // 类型
                Console.WriteLine($"Type: {node.Value.ElementType}");
                Console.WriteLine("------------------------");
            }
        }
    }
}
