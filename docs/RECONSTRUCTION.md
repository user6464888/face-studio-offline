# 可选重建设置

本地重建需要可选工具和权重，不能只安装演示依赖就处理录像。当前自动回归不包含完整录像重建；以下说明资源接口与运行条件。

## 输入与命令

使用主录像＋顶部补拍。主录像应从正脸开始，缓慢覆盖两侧和后脑，表情、曝光及灯光尽量稳定；顶部段须与主段有重叠。两段必须同类：Gemini `.bag`，或同一普通相机、相同分辨率的 RGB 视频。仅正面录像不能恢复完整后脑。

在仓库根目录安装 `requirements-reconstruction.txt`。先运行不写入重建状态的检查：

```powershell
python main.py --reconstruct --primary "D:\my_capture\main.bag" --top "D:\my_capture\top.bag" --check
```

补齐缺少的资源后移除 `--check`。需要完全重算时加 `--force`；默认以文件内容边缘、大小及时间等信息匹配已有结果。新结果写入 `head_reconstruction/runs/`，原输入不覆盖。`--prepare-only` 完成后不启动页面。完整日志和失败阶段留在本机运行目录，可用同样输入续跑。

## Gemini RGB-D 路线

1. 当前采集／播放桥接仅支持 **Windows x64，Orbbec SDK 1.8.3**。使用匹配的 SDK 完整运行库目录，而不是单独拷一个 DLL。
2. 从 [Orbbec 官方仓库](https://github.com/orbbec/OrbbecSDK) 获取兼容版本，阅读其许可。通过环境变量指定 DLL 所在目录：

   ```powershell
   $env:ORBBEC_SDK_DIR="D:\tools\OrbbecSDK\lib\win_x64"
   ```

   不设置时使用 `rgbd_capture/tools/gemini_sdk/OrbbecSDK.dll`。SDK 文件不会被提交或打入发布包。
3. 将 OpenCV YuNet 的 `face_detection_yunet_2023mar.onnx` 放为 `head_reconstruction/models/yunet_2023mar.onnx`。来源：[OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。检查文件实际内容，不能把 Git LFS 指针当成模型。
4. 已有 BAG 可不连接相机播放；新录制可运行 `rgbd_capture/gemini_capture.py`。采集文件及相机侧车信息属于本地数据，发布工具不收录。

深度和 RGB 标定不匹配、黑发深度缺失、转动过快、眼镜反光和顶部重叠不足都会影响重建。补洞和头皮是估计，不代表额外的实测深度。

## 普通 RGB 路线

需要 Torch、ONNX Runtime、分割权重、稀疏特征权重以及 COLMAP 工具。CPU ONNX Runtime 是当前依赖默认项；GPU 版本需自行匹配 CUDA，不能与 CPU 包同时混装后假定会自动使用显卡。

当前源码期待：

| 位置 | 资源 |
|---|---|
| `head_reconstruction/models/birefnet-lite.onnx` | 与分割输入／输出接口兼容的 BiRefNet Lite ONNX |
| `head_reconstruction/models/aliked-n16rot.onnx` | COLMAP 对应构建支持的 ALIKED 模型 |
| `head_reconstruction/models/aliked-lightglue.onnx` | 对应的匹配模型 |
| `head_reconstruction/models/yunet_2023mar.onnx` | YuNet 检测器 |
| `head_reconstruction/tools/colmap/bin/colmap.exe` | 支持现有 ALIKED／LightGlue 参数的 COLMAP 构建 |
| `head_reconstruction/tools/colmap-3.11.1/bin/colmap.exe` | 原稠密重建接口使用的 COLMAP 3.11.1 构建 |

可用 `COLMAP_EXE` 和 `COLMAP_DENSE_EXE` 覆盖两个可执行文件路径。不同 COLMAP 版本的特征参数会变化，需核对兼容性；此版本不提供自动安装器。参考 [COLMAP](https://github.com/colmap/colmap)、[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)、[ALIKED](https://github.com/Shiaoming/ALIKED)、[LightGlue](https://github.com/cvg/LightGlue)。具体 ONNX 转换版本还需与代码接口匹配，不能保证任意同名权重可用。

3DDFA/BFM 人脸先验是历史上的可选增强：只有指定文件齐全时才启用；不随仓库分发，也不是演示的依赖。没有这些资源时跳过增强，不能把 RGB 模型描述为传感器实测深度或精确公制尺寸。

## 从已有重建资产接入

若已通过本项目得到一个完整 assets 目录，可在 `head_reconstruction` 下运行：

```powershell
python -m salon.prepare --assets "D:\my_result\preview\assets\rgbd_b" --heads-only
```

该目录需包含原始网格、已修正光头／纹理、耳部掩码、`styling.json` 和 `metadata.json` 等配套文件，不能只给一个任意 PLY。随后用下面命令打开当前已准备状态；这不会切回示意演示：

```powershell
python -m salon.serve_current --port 8770
```

页面中的真人脸型是几何初判。可通过切换七种示意头比较其他脸型；它不具备人群标注验证的准确率，也没有可靠的颅骨形态分类。
