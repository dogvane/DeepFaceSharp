# DeepFaceSharp

[en](README.en.md) | [中文](README.md)

DeepFaceSharp是一个基于C#的人脸识别库，它是对[deepface](https://github.com/serengil/deepface)的C#实现。

项目的模型来自于[deepface](https://github.com/serengil/deepface)里使用的模型，为了在.net下使用，已经将模型都转为了onnx格式。
模型可以自动从网络下载 [github](https://github.com/dogvane/DeepFaceSharp/releases/tag/v0.0.0)

图片的处理，使用的是OpenCVSharp。

目前移植的人脸检测算法有

- yunet
- yolo8
- yolo11

人脸识别算法有

- arcface
- facenet

项目可以直接使用上述的特定算法，也可以基于 DeepFace 类使用。

## 使用用例

### 人脸检测算法

#### yunet

```csharp
    var yunet = new YuNet();
    var fileName = "../../../../SampleImages/000000001000.jpg";

    var image = Cv2.ImRead(fileName);
    var ret = yunet.DetectFaces(image);
    Console.WriteLine("find face: " + ret.Count);
    foreach (var face in ret)
    {
        DrawFace(image, face);
    }

    Cv2.ImWrite(new FileInfo(fileName).Name.Replace(".jpg", "_YuNet.jpg"), image);
```

#### yolo8

```csharp
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
```

#### yolo11

- 注意：yolo11 算法的返回不带人脸的5点数据

```csharp
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

    Cv2.ImWrite(new FileInfo(fileName).Name.Replace(".jpg", "_yolo11.jpg"), image);
```

### 人脸识别算法

#### arcface

```csharp
    var arcFace = new ArcFace();
    var fileName = "../../../../SampleImages/000000001000_align_False_4.jpg";

    var image = Cv2.ImRead(fileName);
    var ret = arcFace.GetEmbeddingByMat(image);
    Console.WriteLine(ret.Length);
    var txt = string.Join("\n", ret);
    File.WriteAllText("arcface_net_onnx.txt", txt);
```

#### facenet

```csharp
    var facenet = new Facenet();
    var fileName = "../../../../SampleImages/000000001000_align_False_4.jpg";

    var image = Cv2.ImRead(fileName);
    var ret = facenet.GetEmbeddingByMat(image);
    Console.WriteLine(ret.Length);
    var txt = string.Join("\n", ret);
    File.WriteAllText("facenet128_net_onnx.txt", txt);
```

## GPU 版本的适配

虽然onnx不同版本的cuda运行库，支持从 11.8 到 12.x [说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。
但是，.net编译的版本，则没有这么宽泛的版本支持，从 1.20.0 开始, cuda的版本就只支持12.x的了。并且还需 cudnn9 以上的版本。

| ONNX Runtime 版本 |  CUDA 版本 | 备注 |
| ----------------- | --------------- | ---- |
| 1.16.x - 1.19.x   | CUDA 11.8       | 需要 cudnn8 |
| 1.20.0 及以上     | CUDA 12.x       | 需要 cudnn9 |

- 如果发现cuda无法开启，建议根据上述的版本对应关系，以及本机的cuda环境，选择合适的onnx版本。或者退回cpu版本。
- 个人推荐在window环境下使用 DirectML 的版本。

```csharp
    // 启用gpu的方法
    ModelConfiguration.Instance.SetGPUConfig(GPUBackend.DirectML);
```