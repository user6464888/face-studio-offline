# Face Studio · 离线人头重建与发型试戴

一个已结项的研究原型：从本地视频构建人头、估计光头底模，以三层程序化头发展示不同款式。演示包含 **19 款发型、7 种程序化示意脸型**，无需摄像头、私人录像或模型权重即可启动。

![原创示意头上的发型演示](docs/images/demo.png)

**状态：v0.1.0 研究原型，停止主动功能开发，不保证持续维护。** 当前发型为程序化近似，脸型建议为手工规则，均没有经过人群数据验证；不是照片级复刻或理发效果承诺。

## 先了解项目

| 想了解的内容 | 文档 |
|---|---|
| 项目目标、完整流程、输入输出、实现方法和边界 | [完整介绍](docs/PROJECT_OVERVIEW.md) |
| 更自然的标准头从哪里下载、如何适配 | [标准头获取与替换](docs/STANDARD_HEADS.md) |
| 使用什么相机、怎样拍、哪些角度必需 | [摄像头与采集指南](docs/CAPTURE_GUIDE.md) |
| 实际效果图、历史结果和展示图的选择 | [效果实例说明](docs/EXAMPLES.md) |
| 完整重建需要安装哪些可选资源 | [重建设置](docs/RECONSTRUCTION.md) |
| 哪些检查实际通过，哪些尚未验证 | [验证范围](docs/VALIDATION.md) |

本项目开发期间实际使用 **Orbbec Gemini 2、Orbbec SDK 1.8.3、USB 3.0**；两次采集记录均为彩色 1920×1080 @30 FPS、深度 1280×800 @30 FPS。上述设备配置来自本地采集记录，不代表项目已兼容所有深度相机。普通电脑摄像头可提供 RGB 视频，但没有同等的实测深度。

## 快速体验

已验证环境：**Windows x64、CPython 3.11.7**。使用支持 WebGL2 的桌面浏览器。演示不需要 CUDA；三维显示仍需要浏览器图形加速。其他系统未验证；Gemini 的 ctypes SDK 接口目前仅支持 Windows。

在 VS Code 打开本仓库根目录，在终端依次运行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe main.py
```

首次安装需要网络，安装后生成、展示都在本机进行。浏览器会打开 `http://127.0.0.1:8770/salon/index.html`。端口被占用可运行 `main.py --port 8771`；只使用终端输出网址可加 `--no-open`。在 VS Code 选择 `.venv` 的解释器后，也可直接运行 `main.py` 或按 F5。

从源码生成七个示意头；选择一个未缓存的头模／发型组合时再生成三层几何，首次可能需要等待。高密度发型含数百万三角面，会使用较多内存、浏览器显存及磁盘；按需生成的缓存可重新计算，无需提交到仓库。用终端 `Ctrl+C` 停止服务。

若要复现本次 Windows 环境的全部依赖版本，可将上面的安装文件换为 `requirements-demo-win-py311.lock.txt`。它是固定平台快照，不是跨平台锁文件。

## 演示内容

- 七种原创程序化示意头：鹅蛋、圆、方、长、心形、瓜子、菱形。它们来自同一个数学头模的区域变形，不是七位真人扫描，也不是脸型分类标准。
- 当前保留的 19 款沿用历史编号：01–16、19、22、27。每个示意头都能试戴全部款式；推荐不限制选择。
- 三层结构：贴头短发曲面、中长度三维发束、外层长发丝。09 光头渐变仅短层；10 Buzz Cut 短寸为短＋中层；其余款式为三层。
- 发层独立开关、发色、正侧后顶视角、拖动旋转、自动旋转和 PNG 保存。
- 款式介绍、适合脸型和日常打理建议。参考照片未随仓库发布。

默认演示仅按你选择的示意脸型给建议，不会假装扫描了使用者。无需额外学习网络或访问外部图像服务。

## 使用自己的录像重建（可选、实验性）

完整流程是 **主录像＋顶部补拍 → 原始脸模 → 光头底模 → 可更正的几何脸型初判 → 规则推荐 → 更换展示**。主录像须覆盖两侧及后脑；一段静止正脸视频无法恢复完整人头。

完整重建需要另装依赖、相应 SDK／工具和权重；这些文件及私人采集结果均不随仓库分发。先阅读 [重建设置](docs/RECONSTRUCTION.md)。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-reconstruction.txt
.\.venv\Scripts\python.exe main.py --reconstruct --primary "D:\my_capture\main.bag" --top "D:\my_capture\top.bag" --check
.\.venv\Scripts\python.exe main.py --reconstruct --primary "D:\my_capture\main.bag" --top "D:\my_capture\top.bag"
```

也可提供同一相机、相同分辨率的两段 RGB 视频。RGB 路线不具备传感器实测深度或可靠的绝对尺寸；效果对视角、光照、遮挡和运动敏感。重建成功后，本次真人模型会成为当前页面的本地可选项。再次默认运行演示会切回七个示意头；已有重建与录像仍保存在本地，重新使用相同输入可复用重建结果。

项目包含两位参与者的 Gemini 重建实验。仓库不附带参与者数据；当前自动检查覆盖环境安装、演示启动、几何贴合与展示功能，不包含完整录像重建的回归测试。各项实测范围见 [验证记录](docs/VALIDATION.md)。

## 方法与限制

深度录像路线读取 RGB、深度和标定，进行筛帧、配准与融合；顶部补拍提供额外覆盖，缺失区域可能仍需估计。光头处理保留脸部和耳部，估计隐藏头皮的形状与肤色。发根按底模实际三角面贴合，三层头发按款式发流生成。

脸型初判比较正面网格的脸长、额宽、颧宽、下颌和下巴比例；它不是高低颅顶、扁后脑等颅骨形态的可靠分类，也没有标注人群上的准确率。建议仅用于款式比较。

本版没有接入发型 GIF／GLB 导出，也不包含照片级发丝物理模拟。程序化示意头用于方法展示，不作为解剖标准。几何检查通过不代表视觉真实度。

## 目录与检查

```text
main.py                       默认演示／显式重建入口
requirements-demo.txt         演示直接依赖
requirements-reconstruction.txt  可选重建依赖
head_reconstruction/salon/    发型规则、数学头模、几何生成、离线页面
head_reconstruction/viewer/vendor/  Three.js 及原许可
rgbd_capture/                 Gemini 采集和 SDK 桥接源码
head_capture/                 普通摄像头采集源码
docs/                         资源来源、安装、验证、发布说明
scripts/                      发布包生成与检查
tests/                        回归检查
```

```powershell
.\.venv\Scripts\python.exe main.py --check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\build_release.py
```

生成的源码 ZIP 在 `dist/`，采用明确的文件范围，不收录模型缓存、私人数据、安装环境或工具二进制。不要直接把运行后的整个工作文件夹压缩上传。

## 资源与许可

本项目原创代码和新生成的数学示意头使用 [MIT License](LICENSE)。捆绑的 Three.js 文件保留其 MIT 许可证。第三方数据集、模型权重和真人采集数据不随仓库提供，其许可独立于项目源码。

第三方研究来源、运行时依赖及可选权重／SDK 的边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。若另行添加自己的素材，分发权限由素材本身决定。
