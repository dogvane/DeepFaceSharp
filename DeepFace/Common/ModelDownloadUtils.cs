using System.Net;
using System.Net.Http;
using System.Security.Cryptography;

namespace DeepFace.Common
{
    /// <summary>
    /// 模型下载工具类，提供模型的下载与校验功能
    /// </summary>
    public static class ModelDownloadUtils
    {
        /// <summary>
        /// 是否启用默认的进度显示
        /// </summary>
        public static bool EnableDefaultProgressDisplay { get; set; } = true;

        /// <summary>
        /// 代理服务器地址，格式如："http://proxy.example.com:8080"
        /// 如果为空或null则不使用代理
        /// </summary>
        public static string ProxyServer { get; set; } = null;
        
        /// <summary>
        /// 下载重试次数
        /// </summary>
        public static int MaxRetryCount { get; set; } = 3;
        
        /// <summary>
        /// 下载超时时间(秒)
        /// </summary>
        public static int TimeoutSeconds { get; set; } = 600;

        /// <summary>
        /// 创建默认的控制台进度显示器
        /// </summary>
        /// <returns>进度报告接口</returns>
        public static IProgress<double> CreateDefaultProgressHandler()
        {
            // 上一次显示的进度文本长度
            int lastProgressTextLength = 0;
            
            return new Progress<double>(percentage =>
            {
                if (percentage <= 100) 
                {
                    // 构建进度文本
                    string progressText = $"Download Progress: {Math.Floor(percentage)}%"; // 下载进度
                    
                    // 清除上一次显示的文本
                    if (lastProgressTextLength > 0)
                    {
                        Console.Write("\r" + new string(' ', lastProgressTextLength) + "\r");
                    }
                    
                    // 显示新的进度
                    Console.Write("\r" + progressText);
                    
                    // 保存当前文本长度
                    lastProgressTextLength = progressText.Length;
                }

                // 如果下载完成，添加换行符
                if (percentage >= 100)
                {
                    Console.WriteLine();
                    lastProgressTextLength = 0; // 重置长度，防止影响后续输出
                }
            });
        }

        /// <summary>
        /// 从网络下载模型（同步方法）
        /// </summary>
        /// <param name="url">模型URL</param>
        /// <param name="savePath">保存路径</param>
        /// <param name="progress">下载进度回调</param>
        /// <param name="expectedMd5">预期的MD5值（可选，用于校验）</param>
        /// <returns>保存的文件路径</returns>
        public static string DownloadModel(string url, string savePath, IProgress<double> progress = null, string expectedMd5 = null)
        {
            // 确保目录存在
            Directory.CreateDirectory(Path.GetDirectoryName(savePath));

            // 如果没有提供进度处理器，且启用了默认进度显示，创建一个默认的控制台进度显示
            if (progress == null && EnableDefaultProgressDisplay)
            {
                progress = CreateDefaultProgressHandler();
            }

            // 创建HttpClientHandler并配置
            var handler = new HttpClientHandler
            {
                ServerCertificateCustomValidationCallback = (sender, cert, chain, sslPolicyErrors) => true // 允许所有证书(不推荐用于生产环境)
            };
            
            // 如果配置了代理服务器，则设置代理
            if (!string.IsNullOrEmpty(ProxyServer))
            {
                handler.Proxy = new WebProxy(ProxyServer);
                Console.WriteLine($"Using proxy server: {ProxyServer}"); // 使用代理服务器
            }
            
            // 设置最新的TLS版本
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12 | SecurityProtocolType.Tls13;
            
            Exception lastException = null;
            
            // 重试机制
            for (int retry = 0; retry < MaxRetryCount; retry++)
            {
                try
                {
                    if (retry > 0)
                    {
                        Console.WriteLine($"Retry download attempt {retry}..."); // 第 x 次重试下载
                    }
                    
                    using (var httpClient = new HttpClient(handler))
                    {
                        httpClient.Timeout = TimeSpan.FromSeconds(TimeoutSeconds);
                        Console.WriteLine($"Starting model download: {url}"); // 开始下载模型
                        
                        // 异步转同步下载
                        using (var response = httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).Result)
                        {
                            response.EnsureSuccessStatusCode();
                            
                            var totalBytes = response.Content.Headers.ContentLength ?? -1L;
                            
                            using (var contentStream = response.Content.ReadAsStreamAsync().Result)
                            using (var fileStream = new FileStream(savePath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true))
                            {
                                var buffer = new byte[8192];
                                var totalBytesRead = 0L;
                                var bytesRead = 0;
                                
                                while ((bytesRead = contentStream.Read(buffer, 0, buffer.Length)) > 0)
                                {
                                    fileStream.Write(buffer, 0, bytesRead);
                                    totalBytesRead += bytesRead;
                                    
                                    if (progress != null && totalBytes > 0)
                                    {
                                        var progressPercentage = (double)totalBytesRead / totalBytes * 100;
                                        progress.Report(progressPercentage);
                                    }
                                }
                            }
                        }
                        
                        // 确保进度显示完成后再输出其他消息
                        if (progress != null)
                        {
                            progress.Report(100.0); // 确保显示100%
                            Thread.Sleep(100); // 给进度显示一个小延迟
                        }
                        
                        Console.WriteLine("Model download completed"); // 模型下载完成
                        
                        // 如果提供了MD5值，进行校验
                        if (!string.IsNullOrEmpty(expectedMd5))
                        {
                            string actualMd5 = CalculateFileMd5(savePath);
                            if (!string.Equals(actualMd5, expectedMd5, StringComparison.OrdinalIgnoreCase))
                            {
                                File.Delete(savePath); // 校验失败删除文件，准备重试
                                throw new InvalidDataException($"File verification failed. Expected MD5: {expectedMd5}, Actual MD5: {actualMd5}"); // 文件校验失败。预期MD5: {expectedMd5}, 实际MD5: {actualMd5}
                            }
                            Console.WriteLine("File verification passed"); // 文件校验通过
                        }
                        
                        return savePath; // 下载成功，返回
                    }
                }
                catch (Exception ex)
                {
                    lastException = ex;
                    Console.WriteLine($"Download failed: {ex.Message}"); // 下载失败
                    
                    if (ex.InnerException != null)
                    {
                        Console.WriteLine($"Inner error: {ex.InnerException.Message}"); // 内部错误
                    }
                    
                    if (retry == MaxRetryCount - 1)
                    {
                        throw new FileNotFoundException($"Failed to download model file: {ex.Message}", ex); // 下载模型文件失败
                    }
                    
                    // 指数退避策略
                    Thread.Sleep(1000 * (int)Math.Pow(2, retry));
                }
            }
            
            throw new FileNotFoundException("Failed to download model file, exceeded maximum retry attempts", lastException); // 下载模型文件失败，已超过最大重试次数
        }

        /// <summary>
        /// 从网络下载模型（异步方法）
        /// </summary>
        /// <param name="url">模型URL</param>
        /// <param name="savePath">保存路径</param>
        /// <param name="progress">下载进度回调</param>
        /// <param name="expectedMd5">预期的MD5值（可选，用于校验）</param>
        /// <param name="cancellationToken">取消令牌</param>
        /// <returns>保存的文件路径</returns>
        public static async Task<string> DownloadModelAsync(string url, string savePath, IProgress<double> progress = null,
            string expectedMd5 = null, CancellationToken cancellationToken = default)
        {
            // 使用TaskCompletionSource包装同步方法为异步方法
            var tcs = new TaskCompletionSource<string>();

            ThreadPool.QueueUserWorkItem(_ =>
            {
                try
                {
                    var result = DownloadModel(url, savePath, progress, expectedMd5);
                    tcs.SetResult(result);
                }
                catch (Exception ex)
                {
                    tcs.SetException(ex);
                }
            });

            using (cancellationToken.Register(() => tcs.TrySetCanceled()))
            {
                return await tcs.Task;
            }
        }

        /// <summary>
        /// 计算文件的MD5哈希值
        /// </summary>
        public static string CalculateFileMd5(string filePath)
        {
            using (var md5 = MD5.Create())
            using (var stream = File.OpenRead(filePath))
            {
                byte[] hash = md5.ComputeHash(stream);
                return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
            }
        }

        /// <summary>
        /// 检查模型文件是否存在，不存在则从网络下载（同步方法）
        /// </summary>
        public static string EnsureModelExists(string localPath, string downloadUrl,
            IProgress<double> progress = null, string expectedMd5 = null)
        {
            if (File.Exists(localPath))
            {
                // 如果提供了MD5值，验证现有文件
                if (!string.IsNullOrEmpty(expectedMd5))
                {
                    string actualMd5 = CalculateFileMd5(localPath);
                    if (string.Equals(actualMd5, expectedMd5, StringComparison.OrdinalIgnoreCase))
                    {
                        Console.WriteLine("Model file exists and verification passed"); // 模型文件已存在且校验通过
                        return localPath; // 文件已存在且校验通过
                    }
                    // 校验失败，将重新下载
                    Console.WriteLine($"Model file verification failed, re-downloading. Expected MD5: {expectedMd5}, Actual MD5: {actualMd5}"); // 模型文件校验失败，将重新下载。预期MD5: {expectedMd5}, 实际MD5: {actualMd5}
                }
                else
                {
                    Console.WriteLine("Model file already exists"); // 模型文件已存在
                    return localPath; // 文件已存在，无需下载
                }
            }
            else
            {
                Console.WriteLine("Model file not found, downloading required"); // 模型文件不存在，需要下载
            }

            // 下载模型
            return DownloadModel(downloadUrl, localPath, progress, expectedMd5);
        }

        /// <summary>
        /// 检查模型文件是否存在，不存在则从网络下载（异步方法）
        /// </summary>
        public static async Task<string> EnsureModelExistsAsync(string localPath, string downloadUrl,
            IProgress<double> progress = null, string expectedMd5 = null, CancellationToken cancellationToken = default)
        {
            return await Task.Run(() => EnsureModelExists(localPath, downloadUrl, progress, expectedMd5));
        }
    }
}
