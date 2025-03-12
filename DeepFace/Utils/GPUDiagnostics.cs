using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Management; // 添加System.Management命名空间
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DeepFace.Utils
{
    /// <summary>
    /// 提供GPU诊断功能的工具类
    /// </summary>
    public static class GPUDiagnostics
    {
        /// <summary>
        /// 获取可用设备ID和名称信息
        /// </summary>
        /// <returns>包含提供程序和其可用设备ID及名称的字典</returns>
        public static Dictionary<string, List<(int DeviceId, string DeviceName)>> GetAvailableDeviceIds()
        {
            var result = new Dictionary<string, List<(int DeviceId, string DeviceName)>>();
            
            // 初始化结果集
            result["CUDA"] = new List<(int, string)>();
            result["DirectML"] = new List<(int, string)>();
            
            try
            {
                // 只在Windows环境下查询GPU信息
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    using (var searcher = new ManagementObjectSearcher("SELECT * FROM Win32_VideoController"))
                    {
                        int gpuIndex = 0;
                        foreach (ManagementObject obj in searcher.Get())
                        {
                            string gpuName = obj["Name"]?.ToString() ?? $"Unknown GPU #{gpuIndex}";
                            string description = obj["Description"]?.ToString() ?? "";
                            
                            // 判断GPU类型
                            bool isNvidia = gpuName.ToLower().Contains("nvidia") || description.ToLower().Contains("nvidia");
                            bool isAmd = gpuName.ToLower().Contains("amd") || description.ToLower().Contains("amd") || 
                                        gpuName.ToLower().Contains("radeon") || description.ToLower().Contains("radeon");
                            bool isIntel = gpuName.ToLower().Contains("intel") || description.ToLower().Contains("intel");
                            
                            // NVIDIA GPU一般支持CUDA
                            if (isNvidia)
                            {
                                result["CUDA"].Add((gpuIndex, gpuName));
                            }
                            
                            // 所有GPU都可以尝试DirectML (包括NVIDIA, AMD和Intel)
                            if (isNvidia || isAmd || isIntel || !string.IsNullOrEmpty(gpuName))
                            {
                                result["DirectML"].Add((gpuIndex, gpuName));
                            }
                            
                            gpuIndex++;
                        }
                    }
                }                
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"获取GPU设备信息失败: {ex.Message}");
            }
            
            // 如果没有找到任何设备，至少为CPU执行提供一个默认设备
            if (result["CUDA"].Count == 0 && result["DirectML"].Count == 0)
            {
                result["CPU"] = new List<(int, string)> { (0, "CPU Default Device") };
            }
            
            return result;
        }

        /// <summary>
        /// 运行GPU诊断并返回诊断结果
        /// </summary>
        /// <returns>包含诊断信息的字符串</returns>
        public static string RunDiagnostics()
        {
            var sb = new StringBuilder();
            sb.AppendLine("=== GPU 诊断信息 ===");

            // 系统信息
            sb.AppendLine($"操作系统: {Environment.OSVersion}");
            sb.AppendLine($"64位操作系统: {Environment.Is64BitOperatingSystem}");
            sb.AppendLine($"64位进程: {Environment.Is64BitProcess}");
            sb.AppendLine($"处理器数量: {Environment.ProcessorCount}");

            // ONNX Runtime信息
            sb.AppendLine($"ONNX Runtime版本: {typeof(InferenceSession).Assembly.GetName().Version}");

            // 执行提供程序信息
            try
            {
                var availableProviders = OrtEnv.Instance().GetAvailableProviders();
                sb.AppendLine($"可用执行提供程序: {string.Join(", ", availableProviders)}");

                // 获取并显示可用设备ID信息
                var deviceIds = GetAvailableDeviceIds();
                sb.AppendLine("\n可用设备ID信息:");
                foreach (var provider in deviceIds)
                {
                    sb.AppendLine($"  {provider.Key} 提供程序可用设备:");
                    foreach (var device in provider.Value)
                    {
                        sb.AppendLine($"    设备ID: {device.DeviceId}, 设备名称: {device.DeviceName}");
                    }
                }
            }
            catch (Exception ex)
            {
                sb.AppendLine($"获取执行提供程序时出错: {ex.Message}");
            }

            // DLL检查
            var executingPath = AppDomain.CurrentDomain.BaseDirectory;
            sb.AppendLine($"当前执行路径: {executingPath}");
            var requiredDlls = new[] {
                "onnxruntime.dll",
                "onnxruntime_providers_cuda.dll",
                "onnxruntime_providers_shared.dll",
                "onnxruntime_providers_dml.dll",
                "DirectML.dll",
                "cublas64_*.dll",
                "cudnn64_*.dll",
                "cudart64_*.dll",
                "cufft64_*.dll",
                "cublasLt64_*.dll",
                "zlibwapi.dll",
                "dxil.dll",
                "nvrtc64_*.dll",
                "nvrtc-builtins64_*.dll"
            };

            sb.AppendLine("DLL检查:");
            foreach (var dllPattern in requiredDlls)
            {
                if (dllPattern.Contains("*"))
                {
                    var fileNameWithoutExt = Path.GetFileNameWithoutExtension(dllPattern);
                    var extension = Path.GetExtension(dllPattern);
                    var pattern = fileNameWithoutExt.Replace("*", "") + "*" + extension;
                    var matchingFiles = Directory.GetFiles(executingPath, pattern);

                    if (matchingFiles.Length > 0)
                    {
                        sb.AppendLine($"  找到匹配 {dllPattern} 的文件: {string.Join(", ", matchingFiles.Select(Path.GetFileName))}");
                        
                        // 添加版本信息检查
                        foreach (var file in matchingFiles)
                        {
                            try
                            {
                                var versionInfo = FileVersionInfo.GetVersionInfo(file);
                                sb.AppendLine($"    - {Path.GetFileName(file)} 版本: {versionInfo.FileVersion}, 产品版本: {versionInfo.ProductVersion}");
                            }
                            catch
                            {
                                sb.AppendLine($"    - {Path.GetFileName(file)} 无法获取版本信息");
                            }
                        }
                    }
                    else
                    {
                        sb.AppendLine($"  未找到匹配 {dllPattern} 的文件");
                    }
                }
                else
                {
                    var dllPath = Path.Combine(executingPath, dllPattern);
                    sb.AppendLine($"  {dllPattern}: {(File.Exists(dllPath) ? "已找到" : "未找到")}");
                }
            }

            // 尝试获取GPU信息
            try
            {
                // 使用外部命令获取GPU信息
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    var gpuInfo = RunProcessAndGetOutput("powershell", "-Command \"Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | Format-List\"");
                    sb.AppendLine("\nGPU信息 (Windows):");
                    sb.AppendLine(gpuInfo);
                }
            }
            catch (Exception ex)
            {
                sb.AppendLine($"\n获取GPU信息时出错: {ex.Message}");
            }

            // 添加详细GPU信息(Windows)
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                try
                {
                    string detailedGpuInfo = GetWindowsGpuInfo();
                    sb.AppendLine("\n详细GPU硬件信息 (Windows):");
                    sb.AppendLine(detailedGpuInfo);
                }
                catch (Exception ex)
                {
                    sb.AppendLine($"\n获取详细GPU信息时出错: {ex.Message}");
                }
            }

            // 尝试检测CUDA路径
            try
            {
                var envPaths = new[] { "CUDA_PATH", "CUDA_HOME" };
                sb.AppendLine("\nCUDA路径检测:");
                foreach (var envPath in envPaths)
                {
                    var path = Environment.GetEnvironmentVariable(envPath);
                    sb.AppendLine($"  {envPath}: {(string.IsNullOrEmpty(path) ? "未设置" : path)}");
                }
            }
            catch (Exception ex)
            {
                sb.AppendLine($"检测CUDA路径时出错: {ex.Message}");
            }

            return sb.ToString();
        }

        /// <summary>
        /// 获取Windows系统下GPU设备的详细信息
        /// </summary>
        /// <returns>包含GPU设备详细信息的字符串</returns>
        public static string GetWindowsGpuInfo()
        {
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return "此函数仅支持Windows操作系统";
            }

            var sb = new StringBuilder();
            try
            {
                using (var searcher = new ManagementObjectSearcher("SELECT * FROM Win32_VideoController"))
                {
                    int gpuIndex = 0;
                    foreach (ManagementObject obj in searcher.Get())
                    {
                        sb.AppendLine($"--- GPU #{++gpuIndex} ---");

                        // 基本信息
                        AppendPropertyValue(sb, obj, "设备名称", "Name");
                        AppendPropertyValue(sb, obj, "描述", "Description");
                        AppendPropertyValue(sb, obj, "状态", "Status");
                        AppendPropertyValue(sb, obj, "显卡制造商", "AdapterCompatibility");
                        AppendPropertyValue(sb, obj, "设备ID", "DeviceID");

                        // 驱动信息
                        AppendPropertyValue(sb, obj, "驱动版本", "DriverVersion");
                        AppendPropertyValue(sb, obj, "驱动日期", "DriverDate");

                        // 内存信息
                        if (obj["AdapterRAM"] != null)
                        {
                            ulong ramBytes = Convert.ToUInt64(obj["AdapterRAM"]);
                            double ramGB = ramBytes / (1024.0 * 1024.0);
                            sb.AppendLine($"显存大小: {ramGB:F2} GB ({ramBytes} 字节)");
                        }

                        sb.AppendLine();
                    }

                    if (gpuIndex == 0)
                    {
                        sb.AppendLine("未找到GPU设备信息");
                    }
                }
            }
            catch (Exception ex)
            {
                sb.AppendLine($"获取GPU信息时发生异常: {ex.Message}");
                if (ex.InnerException != null)
                {
                    sb.AppendLine($"内部异常: {ex.InnerException.Message}");
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// 向StringBuilder添加属性值
        /// </summary>
        private static void AppendPropertyValue(StringBuilder sb, ManagementObject obj, string displayName, string propertyName, string unit = "")
        {
            if (obj[propertyName] != null && !string.IsNullOrEmpty(obj[propertyName].ToString()))
            {
                sb.AppendLine($"{displayName}: {obj[propertyName]}{unit}");
            }
        }

        /// <summary>
        /// 运行外部进程并获取输出
        /// </summary>
        private static string RunProcessAndGetOutput(string fileName, string arguments)
        {
            using (var process = new Process())
            {
                process.StartInfo.FileName = fileName;
                process.StartInfo.Arguments = arguments;
                process.StartInfo.UseShellExecute = false;
                process.StartInfo.RedirectStandardOutput = true;
                process.StartInfo.CreateNoWindow = true;

                process.Start();
                string output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();

                return output;
            }
        }

        /// <summary>
        /// 运行测试推理以验证GPU加速
        /// </summary>
        public static (bool Success, string Provider, TimeSpan Duration) TestInference(string modelPath, bool useCuda = true, bool useDirectML = true)
        {
            try
            {
                if (!File.Exists(modelPath))
                {
                    return (false, "None", TimeSpan.Zero);
                }

                var sessionOptions = new SessionOptions();
                string provider = "CPUExecutionProvider";
                StringBuilder log = new StringBuilder();
                log.AppendLine("推理测试日志:");
                
                // 获取可用的执行提供程序
                var availableProviders = OrtEnv.Instance().GetAvailableProviders();
                log.AppendLine($"系统可用执行提供程序: {string.Join(", ", availableProviders)}");

                if (useCuda && availableProviders.Contains("CUDAExecutionProvider"))
                {
                    try
                    {
                        log.AppendLine("尝试初始化CUDA执行提供程序...");
                        sessionOptions.AppendExecutionProvider_CUDA(0);
                        provider = "CUDAExecutionProvider";
                        log.AppendLine("CUDA执行提供程序初始化成功");
                    }
                    catch (Exception ex)
                    {
                        log.AppendLine($"CUDA初始化失败: {ex.Message}");
                        if (useDirectML && availableProviders.Contains("DmlExecutionProvider"))
                        {
                            try
                            {
                                log.AppendLine("尝试初始化DirectML执行提供程序...");
                                sessionOptions.AppendExecutionProvider_DML(0);
                                provider = "DmlExecutionProvider";
                                log.AppendLine("DirectML执行提供程序初始化成功");
                            }
                            catch (Exception dmlEx)
                            {
                                log.AppendLine($"DirectML初始化失败: {dmlEx.Message}");
                            }
                        }
                    }
                }
                else if (useDirectML && availableProviders.Contains("DmlExecutionProvider"))
                {
                    try
                    {
                        log.AppendLine("尝试初始化DirectML执行提供程序...");
                        sessionOptions.AppendExecutionProvider_DML(0);
                        provider = "DmlExecutionProvider";
                        log.AppendLine("DirectML执行提供程序初始化成功");
                    }
                    catch (Exception ex)
                    {
                        log.AppendLine($"DirectML初始化失败: {ex.Message}");
                    }
                }
                else
                {
                    log.AppendLine($"未找到请求的执行提供程序，将使用CPU执行");
                }

                // 测量创建会话的时间
                var watch = Stopwatch.StartNew();
                using (var _session = new InferenceSession(modelPath, sessionOptions))
                {
                    // 获取实际使用的执行提供程序
                    var actualProvider = _session.InputMetadata;
                }
                watch.Stop();
                
                Console.WriteLine(log.ToString());
                return (true, provider, watch.Elapsed);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"推理测试失败: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"内部异常: {ex.InnerException.Message}");
                }
                return (false, $"Error: {ex.Message}", TimeSpan.Zero);
            }
        }
    }
}
