# 环境搭建

## 系统依赖：ffmpeg

| 平台 | 命令 |
|---|---|
| macOS | `brew install ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` 或 `scoop install ffmpeg` |
| Debian/Ubuntu | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Arch | `sudo pacman -S ffmpeg` |

自制构建（带全部硬件编码器，Linux）：

```bash
sudo apt install ffmpeg   # 发行版包通常已含 h264_nvenc / h264_qsv / h264_amf
ffmpeg -hide_banner -encoders | grep h264   # 确认你的机器有哪个
```

找不到 ffmpeg 时脚本会直接给出上面这行命令，不会静默失败。

## 只在给链接时才需要：yt-dlp

本地文件输入不需要它。给 B站 / YouTube 链接时才用得上：

| 平台 | 命令 |
|---|---|
| macOS | `brew install yt-dlp`（也读得到 brew 的 `libexec` 里的 python） |
| Windows | `winget install yt-dlp` 或 `pip install --user -U yt-dlp` |
| Linux | `pip install --user -U yt-dlp`（或发行版包） |

脚本按这个顺序找它，**每个候选都实际跑一次 `--version` 才算数**（存在但跑不起来
的残缺安装比找不到更难查）：

1. `$DSH_YTDLP_PATH`
2. `PATH`
3. `~/.local/bin`、`/opt/homebrew/bin`、`/usr/local/bin`、
   Windows 的 `%APPDATA%\Python\Python3xx\Scripts`

## cookie：决定能拿到几 P

B站 匿名只给到 720p，1080p 及以上要登录态。cookie 依次取：

1. `--cookies /path/to/cookies.txt`
2. `$DSH_VIDEO_COOKIES`
3. `~/.dsh/video.json` 里的 `cookies` 字段

前两条是为了兼容 DSH 视频插件 —— 它用的就是这套，所以在 DSH 里能正常看 B站 的
机器，装好本 skill 就能直接下 1080p，不用再配一遍。

**自己生成一份 cookies.txt**（Netscape 格式，别用浏览器导出的 JSON）：

```bash
# 有 yt-dlp 的话，最省事的是让它从浏览器读（Chromium 127+ 可能被 App-Bound 加密挡住）
yt-dlp --cookies-from-browser chrome --cookies /tmp/c.txt --skip-download <任意 B站 链接>
chmod 600 /tmp/c.txt

# 被加密挡住时，用浏览器扩展导出 cookies.txt（Netscape 格式），或者手写：
# 至少要 SESSDATA、bili_jct、buvid3 三个字段
```

cookie 会过期。报「需要登录」先去刷新 cookie，别去怀疑代码。

## Python 依赖

```bash
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install numpy opencv-python-headless torch
```

**PyTorch 按平台装**，默认的 `pip install torch` 在很多机器上是不带 GPU 支持的：

| 硬件 | 安装 |
|---|---|
| Apple Silicon | 👆 默认即可（自带 MPS） |
| NVIDIA (CUDA 12.x) | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| AMD ROCm (Linux) | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.0` |
| Intel Arc (Linux/Windows) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` + `pip install intel-extension-for-pytorch` |
| Intel 核显 / 任意 DX12 显卡 (Windows) | `pip install torch-directml`，运行时加 `--device dml` |
| 纯 CPU | 👆 默认即可 |

不想装 torch（只为预览）也可以，但**只有 LaMa 需要 torch**；delogo / alpha 两条路只需要
ffmpeg + numpy + opencv，可以先跑这两条给用户看效果。

## 模型权重

```bash
# 自动下载（脚本内置，会校验 sha256）
python3 -c "import common; print(common.ensure_model('models'))"

# 或手动（205 MB，TorchScript）
curl -L -o models/big-lama.pt \
  https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
```

镜像（GitHub 访问不畅时）：

```bash
curl -L -o models/big-lama.pt \
  https://hf-mirror.com/okaris/simple-lama/resolve/main/big-lama.pt
```

sha256 应为 `344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9`。
脚本校验不过会拒绝使用——**不要绕过校验**，下载被截断的 .pt 在 `torch.jit.load` 里报的错
和「模型不兼容」一模一样，很难查。

## 报错对照

| 现象 | 原因 / 处理 |
|---|---|
| `ffmpeg/ffprobe not found` | 按上面装，或设 `FFMPEG_BIN=/path/to/ffmpeg` |
| `could not determine frame size` | 视频不是标准格式，或 `-pix_fmt` 变了；用 `ffprobe` 确认 |
| `AttributeError: module 'torch' has no attribute 'xpu'` | torch 版本旧，忽略即可（脚本已包在 try 里） |
| `torch.jit.load` 报 unpickling / archive 错 | 权重文件损坏或不是 TorchScript 版；重下并确认 sha256 |
| `mps` 跑起来比 CPU 还慢 | 第一次调用有编译开销；脚本已预热。若持续异常，加 `--device cpu` |
| Windows 上 `dml` 报 device 找不到 | `pip install torch-directml`，且显卡要支持 DX12 |
| 输出文件很大 | 用 `--crf 20` 或硬件编码器 |
| 中文路径报错 | 脚本用列表传参，正常；若仍出错，检查是否用了旧版 Python（<3.8） |

## 性能参考（1080p / 30fps，LaMa 512×512 输入）

| 设备 | 每帧 | 5 分钟片 |
|---|---|---|
| Apple M5 (MPS) | ~0.23 s | ~35 min |
| Apple M1 (MPS) | ~0.35 s | ~50 min |
| RTX 4090 (CUDA) | ~0.04 s | ~6 min |
| RTX 3060 (CUDA) | ~0.07 s | ~10 min |
| Intel Arc A770 (XPU) | ~0.12 s | ~18 min |
| Intel Iris Xe (DirectML) | ~0.5 s | ~70 min |
| CPU (8 核现代 x86) | ~4 s | ~9 h |

以上为非 M5 设备的经验估算，实际以脚本打印的 ETA 为准——**永远用脚本自己的 ETA 回复用户**。
