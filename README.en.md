# DeepFaceSharp

[en](README.en.md) | [中文](README.md)

DeepFaceSharp is a C#-based facial recognition library, which is a C# implementation of [deepface](https://github.com/serengil/deepface).

The models in this project are derived from those used in [deepface](https://github.com/serengil/deepface). To use them in .NET, they have been converted to ONNX format.
Models can be automatically downloaded from [GitHub](https://github.com/dogvane/DeepFaceSharp/releases/tag/v0.0.0).

OpenCVSharp is used for image processing.

Currently, the ported face detection algorithms include:

- yunet
- yolo8
- yolo11

Face recognition algorithms include:

- arcface
- facenet

The project can directly use the specific algorithms mentioned above, or can be used based on the DeepFace class.

## Usage Examples

### Face Detection Algorithms

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

    // Save the marked image
    Cv2.ImWrite(new FileInfo(fileName).Name.Replace(".jpg", "_yolo8.jpg"), image);
```

#### yolo11

- Note: yolo11 algorithm's return does not include the 5-point facial data

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

### Face Recognition Algorithms

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

## GPU Version Adaptation

Although different versions of the ONNX CUDA runtime support from 11.8 to 12.x [details](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
the .NET compiled versions do not have such broad version support. Starting from version 1.20.0, CUDA support is limited to 12.x and requires cudnn9 or above.

| ONNX Runtime Version |  CUDA Version | Notes |
| -------------------- | ------------- | ----- |
| 1.16.x - 1.19.x      | CUDA 11.8     | Requires cudnn8 |
| 1.20.0 and above     | CUDA 12.x     | Requires cudnn9 |

- If CUDA cannot be enabled, it's recommended to choose an appropriate ONNX version according to the version correspondence above and your local CUDA environment, or fall back to the CPU version.
- For Windows environments, I personally recommend using the DirectML version.

```csharp
    // Method to enable GPU
    ModelConfiguration.Instance.SetGPUConfig(GPUBackend.DirectML);
```
