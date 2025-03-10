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