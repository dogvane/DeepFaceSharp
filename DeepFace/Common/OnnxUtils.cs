using DeepFace.Config;
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
        public static InferenceSession CreateSession(string modelPath, int deviceId, GPUBackend preferredBackend)
        {
            try
            {
                var sessionOptions = new SessionOptions();
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

                bool gpuInitialized = false;

                // 根据首选GPU后端尝试初始化
                if (preferredBackend == GPUBackend.Auto || preferredBackend == GPUBackend.CUDA)
                {
                    try
                    {
                        // 尝试CUDA
                        //var availableProviders = OrtEnv.Instance().GetAvailableProviders();
                        //if (availableProviders.Contains("CUDAExecutionProvider"))
                        //{
                            sessionOptions.AppendExecutionProvider_CUDA(deviceId);
                            Console.WriteLine($"Using CUDA GPU for inference with device ID: {deviceId}");
                            gpuInitialized = true;
                            return new InferenceSession(modelPath, sessionOptions);
                        //}
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Failed to initialize CUDA: {ex.Message}");
                        if (preferredBackend == GPUBackend.Auto)
                        {
                            Console.WriteLine("Falling back to DirectML...");
                        }
                        else
                        {
                            Console.WriteLine("Falling back to CPU.");
                        }
                    }
                }

                // 如果CUDA未初始化，并且首选后端为Auto或DirectML，则尝试DirectML
                if (!gpuInitialized && (preferredBackend == GPUBackend.Auto || preferredBackend == GPUBackend.DirectML))
                {
                    try
                    {
                        // 检查是否有可用的DirectML执行提供程序
                        //var availableProviders = OrtEnv.Instance().GetAvailableProviders();
                        //if (availableProviders.Contains("DmlExecutionProvider"))
                        //{
                            sessionOptions.AppendExecutionProvider_DML(deviceId);
                            Console.WriteLine($"Using DirectML (GPU) for inference with device ID: {deviceId}");
                            return new InferenceSession(modelPath, sessionOptions);
                        //}
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Failed to initialize DirectML: {ex.Message}");
                        Console.WriteLine("Falling back to CPU.");
                    }
                }

                Console.WriteLine("No GPU acceleration available. Using CPU for inference.");
                return new InferenceSession(modelPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to use GPU: {ex.Message}. Falling back to CPU.");
                return new InferenceSession(modelPath);
            }
        }
    }
}
