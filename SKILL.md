---
name: bilibili-watermark-remover
description: Remove the Bilibili (哔哩哔哩) watermark from a video file. Use ONLY when the user explicitly asks to remove a Bilibili/b站/哔哩哔哩 watermark from their own video. Detects the mark first and refuses to proceed if it is not Bilibili's; offers three removal methods with a rendered comparison image, recommends LaMa, and only renders the full clip after the user picks one. Do not trigger for other watermarks, for stock-footage cleanup, or for generic object removal.
---

# 哔哩哔哩水印去除

## 触发条件（严格）

**只在用户明确要求去掉视频里的哔哩哔哩 / B站水印时启动。** 不因为视频里碰巧有水印就自己开工；
不用于其他平台水印、影视去台标、通用物体擦除、去字幕。用户没说要，就只回答、不动手。

判据是用户的话，不是文件名：出现「去水印」但配的是抖音/YouTube/其他水印 → 不触发；
出现「B站」「哔哩哔哩」「bilibili」+「水印」「去掉」「干净版」→ 触发。

## 这套流程的原则

1. **先证明是 B站的水印，再动手。** B站的上传水印是同一套绘制（UP 主名 + 台标图形），
   位置随上传者/分辨率变化，但物理特征一致：**静态、纯白半透明（α≈0.7）、细笔画、无描边**。
   先跑检测器，通过了才继续。
2. **第一轮只出方案，不出成片。** 除了用户明说「直接出成片」或视频极短（<15 秒），
   第一轮必须：检测 + 一张对比图 + 说明 + 推荐 LaMa + 等用户选。渲染 5 分钟片要 30+ 分钟，
   先斩后奏会浪费用户一晚。
3. **只改掩膜内的像素。** 掩膜外一个字节都不许动，这是「无缝」的唯一保证。
4. **输入是什么就给什么。** 用户甩一个 B站 链接过来时不要反问他「先下载一下」——
   脚本自己会下。只有在他要的清晰度必须登录、而这台机器没有 cookie 时才回头问。

## 工作目录约定

所有中间产物放在视频同目录的 `wm_work/`（或用户指定），最后交付：

```
wm_work/
  report.json          检测报告（含测得的框、alpha、判定）
  mask.png             外扩掩膜（外扩 3px@1080p，用于 LaMa / delogo）
  mask_tight.png       精确字形掩膜（用于 alpha 反解、框计算）
  alpha.npy            逐像素 alpha
  mask_zoom.png        掩膜可视化，给用户看「框对了没」
  resolved.json        仅链接输入时：下载到哪了、标题、时长
```

链接输入的下载件不进 `wm_work/`，而是留在共享缓存
`~/.cache/bilibili-wm-remover/media/<hash>/`（可用 `BILI_WM_MEDIA_DIR` 改）。
这样同一个链接做检测、预览、渲染时只下一次；删不删由用户决定，别自作主张清理。

## 流程

### 第 0 步：环境与输入

```bash
python3 -c "import cv2, numpy, torch; print(torch.__version__, cv2.__version__)"
ffmpeg -version | head -1
```

缺依赖就按平台给一行命令（见 `references/SETUP.md`），别自己闷头装半天。

**不确定这台机器缺什么，就先跑自检**（不下载任何东西，几秒钟）：

```bash
python3 scripts/doctor.py
```

它逐项报 `[OK]/[WARN]/[MISS]`：python / numpy+cv2 / torch 与可用设备 / ffmpeg /
yt-dlp / cookies / 磁盘余量，并在每一项后面直接给出修复命令。**新机器上第一步就跑它**，
比让用户逐个报错快得多。

**输入可以是本地文件，也可以直接是 B站 链接** —— 所有脚本都吃链接，内部会自动下载：

```bash
python3 scripts/detect_watermark.py 'https://www.bilibili.com/video/BV1xx411c7mD' -o wm_work
# 或者先把片子下下来（大文件可以先跑着）
python3 scripts/fetch_source.py 'https://www.bilibili.com/video/BV1xx411c7mD' --height 1080 -o wm_work
```

下载走 `yt-dlp`，落在 `~/.cache/bilibili-wm-remover/media/`（键 = 链接 + 清晰度），
重跑不会重复下。cookie 依次取 `--cookies` → `$DSH_VIDEO_COOKIES` →
`~/.dsh/video.json` 的 `cookies` 字段 —— 和 DSH 的视频插件是同一套来源，
所以装过 DSH 并登录过 B站 的机器不用额外配置。

**没 cookie 也能跑，只是上限 720p**；1080p 及以上需要登录态。用户只给链接时，
先确认清晰度能不能接受，别下完才发现只有 480p。

### 第 1 步：检测 + 校验（约 20–40 秒，与片长无关）

```bash
python3 scripts/detect_watermark.py "$VIDEO" -o wm_work --verbose
```

- 退出码 0 = 判定通过；2 = **不是 B站水印 / 判据不足**（就此停手，不要硬做；
  真要强行处理，加 `--force`，后果自己承担）。
- 判定分两道闸，任何一道不过就是退出码 2：
  **形态闸**——静态性、能量对比、长宽比、占画面比例、右侧密集台标（左侧稀疏文字）、
  且不是"均匀方块"；**物理闸**——解出的 alpha 必须落在 0.42–0.86（B站水印实测 0.62–0.78）。
  不透明（≈0.95）说明是别的工具烙上去的图形，不是 B站水印。
- 用 `mask_zoom.png` 自己看一眼：红色是掩膜，绿框是外扩后的框。**框没包住全文 → 停下来看一眼报告里的 tight_box**，
  必要时用 `--bbox x,y,w,h` 手工指定。
- 记录报告里的 `alpha_median`：本类水印通常在 **0.65–0.72**。明显偏离（<0.4 或 >0.85）说明采样帧背景变化不足，
  或者根本不是这类水印——先复核，别硬信。

**如果检测器判定不通过但用户坚持看到了水印**：让用户用截图指出位置，或自己用
`ffmpeg -ss <t> -i in.mp4 -frames:v 1 probe.png` 抽一帧看，然后 `--bbox` 手工指定重跑。
不要因为脚本说「不是」就断定用户看错了。

### 第 2 步：出对比图（第一轮的核心交付）

```bash
python3 scripts/make_preview.py "$VIDEO" --work wm_work -o preview.png
```

四宫格：原始 / LaMa / delogo / alpha，同一帧、同一区域。把它展示给用户，
告诉他四种方法的代价（见下表），**明确推荐 2 LaMa**，并说明「你点头我就开跑，约 N 分钟」。

**务必点明预览是单帧演示，不是全片承诺**：水印所在的那条带子里，如果背景里本来就有
很亮的细节（高光、白色物体、过曝天空），那一帧的 LaMa 面板会看到淡淡的残影。
脚本会自动挑「周边细节最多」的帧来演示，所以这是最容易露馅的例子，不是平均表现。
成片里这类帧通常只占极少数，但用户应该提前知道有这回事，而不是成片出来才意外。

### 第 3 步：等用户选（除非已被授权直接出片）

用户闲话/没回之前不要开渲染。用户选了就直接进第 4 步；如果用户说「你看着办」→ 用 LaMa。

### 第 4 步：渲染

```bash
# 2 LaMa（推荐）
python3 scripts/remove_lama.py  "$VIDEO" out_lama.mp4   --work wm_work
# 3 delogo（快，但高细节背景会糊）
python3 scripts/remove_delogo.py "$VIDEO" out_delogo.mp4 --work wm_work
# 4 alpha 反解（平缓/暗场可用，亮背景会有鬼影）
python3 scripts/remove_alpha.py "$VIDEO" out_alpha.mp4  --work wm_work
```

长片用后台任务跑，定期回报进度（脚本自带 ETA 输出）。**别忘了 `caffeinate`（macOS）/ 关掉睡眠**，
否则跑一半机器睡了。

### 第 5 步：验收（交付前必做）

```bash
python3 scripts/verify_result.py out_lama.mp4 --src "$VIDEO" --work wm_work
```

- `removed_pct ≥ 85` 且 `verdict: PASS` 才算过。
- 帧数必须与原片一致，音轨必须是原码流（`-c:a copy`，脚本已处理）。
- 不过就回第 1 步查掩膜，不要交付一个「看起来还行」的文件。
- 若 `verify_result.py` 直接报 `FAIL: residual could not be measured`，
  通常是掩膜被存成了 0/1 的 PNG（读回来全黑）。掩膜必须是 0/255 的单通道图。

## 机器没配好时怎么办（这是常态，不是意外）

这个 skill 会被发到完全陌生的机器上：可能没装 torch，可能没有 yt-dlp，可能浏览器里
根本没登录 B站。**不要假装这些不存在，也不要默默降级交一个 720p 的片子**。
正确做法是：说清楚「现在能做什么 / 缺什么 / 补上要哪一条命令」，再让用户决定。

先跑 `python3 scripts/doctor.py`，它会把下面这些一次查完。

### 缺 ffmpeg / numpy / cv2（硬依赖，没有就跑不了）

| 平台 | 命令 |
|---|---|
| macOS | `brew install ffmpeg` / `pip install -r requirements.txt` |
| Windows | `winget install Gyan.FFmpeg` / `pip install -r requirements.txt` |
| Linux | `sudo apt install ffmpeg` / `pip install -r requirements.txt` |

### 缺 torch（只有 2 号 LaMa 需要）

不要因此卡住整个流程。可以先告诉用户：**3 号 delogo 和 4 号 alpha 现在就能跑**，
2 号要装 torch（各平台命令见 `references/SETUP.md`，注意按显卡选对 wheel）。
把选择权交给他：先看 3/4 的效果，还是先装 torch 再比。

### 缺 yt-dlp（只有链接输入需要）

本地文件完全不受影响。给链接时才发现没有，就一次说清：

```bash
# macOS
brew install yt-dlp
# Windows
winget install yt-dlp
# 任意平台（pip）
pip install --user -U yt-dlp
```

装完 `doctor.py` 应显示 `[OK] yt-dlp`。

### 没有 cookie，或者只拿到 720p（最常见的求助）

B站 匿名对多数投稿只给到 720p。**这时候不要直接开渲染**，先问一句：

> 现在能下 720p，去水印效果一样，只是成片是 720p。要 1080p 的话需要登录态，
> 大概一分钟能配好，要现在弄吗？

要的话，按成功率从高到低给三条路：

1. **让脚本从浏览器里读**（最省事，能读就不用导出）

   ```bash
   python3 scripts/doctor.py --from-browser edge     # 或 chrome / firefox / safari
   ```

   成功会打印写到了哪里（默认 `~/.config/bilibili-wm-remover/cookies.txt`，权限 600）。
   失败会明确说原因，不要含糊过去：
   - `could not find X cookies database` → 那个浏览器没装、或没有这个 profile
   - `no bilibili cookies found` → 浏览器里没登录过 bilibili.com，先去登录
   - `App-Bound / decrypt` → Chromium 127+ 的新加密，换 Safari/Firefox，
     或用扩展导出（见下一条）

2. **浏览器扩展导出 cookies.txt**（任何浏览器都行），格式必须是 **Netscape**，不是 JSON。
   然后要么 `--cookies /path/to/cookies.txt`，要么 `export DSH_VIDEO_COOKIES=/path/to/cookies.txt`。

3. **已经在用 DSH 的视频工具** → 什么都不用做。本 skill 先读 `$DSH_VIDEO_COOKIES`，
   再读 `~/.dsh/video.json` 的 `cookies` 字段，最后才看自己的配置目录。

配好之后让他再跑一次 `python3 scripts/doctor.py` 确认 `[OK] cookies ... SESSDATA`，
别让用户以为「导出了就好了」—— 没有 SESSDATA 的文件等于没登录。

### 不要做的事

- 不要用 `sudo`、不要替用户装系统级软件，除非他明确同意；先给命令。
- 不要在用户不知情时把清晰度从 1080 换成 720 然后当成完成。
- 不要把 cookie 内容打印出来、贴进对话或提交到仓库（`.gitignore` 已经排除）。
- 不要因为缺 cookie 就说「这个视频不能处理」—— 720p 一样能去水印。
## 已知的极限（照实说，不要粉饰）

1. **过曝/高光处的残影**。α≈0.7 的白色叠加放在接近 255 的背景上，等于把那块画面削平了；
   LaMa 只能从周围结构去「猜」，猜不准的地方会留下很淡的轮廓。整片里这类帧通常很少，
   但如果本片大量是强逆光、雪地、白墙，要提前说明。
2. **主名特别长或特别小时**，台标签名（right/left 密度比）可能达不到阈值 →
   检测器会判 `validated: false`，此时用 `--bbox` 手工框，并自己看一眼 `mask_zoom.png`。
3. **水印淡出/移动**（极少数上传会做动画）→ 静态假设失效，本流程不适用，应直接告诉用户。

## 三种方法的选择依据

| | 2 LaMa | 3 delogo | 4 alpha |
|---|---|---|---|
| 痕迹 | 极微（细节区略软） | 高细节背景出现横条糊+棋盘格 | 字迹残影 + 彩边（亮背景无解） |
| 画幅 | 不变 | 不变 | 不变 |
| 速度（1080p/30fps，1080p GPU） | 0.06–0.25 s/帧 | ~实时 1–10× | 极快 |
| 适用 | **默认** | 天空/墙面/暗场/虚焦 | 平缓或暗背景；或作为 LaMa 预处理 |

要判断第 3 种在这条片上会不会糊，看 `remove_delogo.py` 打印的
「background detail」：<12 大致安全，≥12 会明显糊（脚本自己会警告）。

## 画质参数的取法（不许拍脑袋写死）

`common.crf_from_source()` 按源码率推：

| 源码 bpp | CRF |
|---|---|
| ≥0.25 | 17 |
| 0.15–0.25 | 18 |
| 0.08–0.15 | 19 |
| <0.08 | 21 |

用户点名要更高画质就 `--crf 16`（文件更大、更慢）；要小文件就 `--crf 20`。
**硬件编码器**：`--encoder h264_videotoolbox`（mac）、`h264_nvenc`（NVIDIA）、
`h264_qsv`（Intel）、`h264_amf`（AMD）。默认 `libx264`，兼容性最好。

## 跨平台要点

- **设备选择**：`common.pick_device()` 依次探测 mps / cuda / xpu / dml / cpu，也可 `--device` 指定。
  Windows 上没装 CUDA 但有 DX12 显卡 → 装 `torch-directml`，用 `--device dml`。
- **Intel 核显**：Windows 走 DirectML；Linux 需要 `intel_extension_for_pytorch`（XPU）。
  两者都没有时 CPU 也能跑，只是慢（1080p 约 4–6 s/帧）。
- **显存/内存**：模型只吃 512×512，峰值占用 <2 GB，8 GB 机器够用。
- **别信 ffprobe 的分辨率**：竖屏视频带 rotation 元数据时，实际解码宽高会互换。
  `common.actual_frame_size()` 会解一帧核对，所有脚本都用它。
- **路径带空格/中文**：全程 `subprocess` 列表传参，没有 shell 拼接，放心。

## 反模式（这些事不要做）

- 检测没过就开工 —— 这是给别人视频加噪点。
- 掩膜整块贴回（而不是只替换掩膜内像素）—— 必留矩形接缝。
- LaMa 输入 512×512 时用黑边 padding —— 会把黑色当真实内容画进洞里。要**拉伸**。
- 掩膜膨胀 15px 以上 —— 浮雕纹理会被整片抹平。默认 3px@1080p。
- 只信肉眼 —— 一定要跑 `verify_result.py` 的数字。
- 对亮背景用 alpha 反解还承诺「能去掉」—— 那里信息已经被削平了，物理上做不到。

## 参考

- `references/SETUP.md`：各平台依赖安装、GPU 后端选择、常见报错
- `references/PITFALLS.md`：踩过的坑（ONNX 输入约定、CoreML 更慢、批量无收益等）
- `references/MEASUREMENTS.md`：在真实 B站片子上测到的水印参数与各方法实测数据
