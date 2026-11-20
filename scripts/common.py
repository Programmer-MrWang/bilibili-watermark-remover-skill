"""Shared helpers for the Bilibili watermark remover.

Deliberately dependency-light and cross-platform:
  - probing / decoding go through ffmpeg only
  - device selection probes mps / cuda / xpu / directml / cpu in that order
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import cv2

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")

# Verified reachable 2026-11; the mirrors are listed so one blocked host is
# not fatal (GitHub releases, then a HuggingFace mirror for mainland China).
MODEL_URLS = [
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
    "https://hf-mirror.com/okaris/simple-lama/resolve/main/big-lama.pt",
]
MODEL_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"
MODEL_NAME = "big-lama.pt"


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def require_ffmpeg():
    missing = [e for e in (FFMPEG, FFPROBE) if shutil.which(e) is None]
    if missing:
        sys.exit(
            "ffmpeg/ffprobe not found: %s\n"
            "  macOS:   brew install ffmpeg\n"
            "  Windows: winget install Gyan.FFmpeg   (or scoop install ffmpeg)\n"
            "  Linux:   sudo apt install ffmpeg      (or dnf/pacman equivalent)\n"
            % ", ".join(missing)
        )


def probe(path):
    """Return width/height/fps/duration/nb_frames/bitrate/audio/rotation."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    a = [s for s in d["streams"] if s["codec_type"] == "audio"]

    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0

    duration = float(d["format"].get("duration") or 0)
    try:
        nb = int(v.get("nb_frames") or 0)
    except ValueError:
        nb = 0
    if nb <= 0:
        nb = int(round(duration * fps))

    bitrate = int(v.get("bit_rate") or d["format"].get("bit_rate") or 0)
    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    for tag in (v.get("tags", {}) or {}):
        if tag == "rotate":
            rotation = int(v["tags"][tag])

    return {
        "path": os.path.abspath(path),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": fps,
        "fps_str": v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1",
        "duration": duration,
        "nb_frames": nb,
        "bitrate": bitrate,
        "has_audio": bool(a),
        "vcodec": v.get("codec_name"),
        "rotation": rotation,
        "size": int(d["format"].get("size") or 0),
    }


def actual_frame_size(path, hint_w, hint_h):
    """Decode one frame and derive the DISPLAYED pixel size.

    ffmpeg autorotates by default, so a 90/270 rotated file decodes with
    swapped dimensions versus what ffprobe reports. Never trust the probe
    for buffer sizing - verify against a real frame.
    """
    buf = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path, "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True, check=True,
    ).stdout
    n = len(buf)
    if n == hint_w * hint_h * 3:
        return hint_w, hint_h
    if n == hint_h * hint_w * 3:
        return hint_h, hint_w
    raise SystemExit(
        "could not determine frame size (%d bytes, probe says %dx%d)" % (n, hint_w, hint_h)
    )



# --------------------------------------------------------------------------- #
# frame IO
# --------------------------------------------------------------------------- #
def iter_sampled(path, want):
    """Yield (\`t\`, raw_bgr_bytes, w, h) for ~\`want\` evenly spaced frames.

    Decoding always runs at full size; callers downscale as they wish. The
    window is killed on early exit so a partial read never leaves a zombie
    ffmpeg pumping frames into a closed pipe.
    """
    p = probe(path)
    w, h = actual_frame_size(path, p["width"], p["height"])
    k = max(1, p["nb_frames"] // max(1, want))
    pr = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", path, "-vf", "select=not(mod(n\\,%d))" % k,
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    buf = w * h * 3
    n = 0
    try:
        while n < want:
            raw = pr.stdout.read(buf)
            if len(raw) < buf:
                break
            yield (n * k) / max(p["fps"], 1e-6), raw, w, h
            n += 1
    finally:
        try:
            pr.stdout.close()
        except Exception:
            pass
        pr.kill()
        pr.wait()


def frames_at(path, times, w, h):
    """Decode one frame per timestamp using fast seeking."""
    for t in times:
        raw = subprocess.run(
            [FFMPEG, "-v", "error", "-ss", "%.3f" % t, "-i", path, "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True, check=True,
        ).stdout
        if len(raw) == w * h * 3:
            yield t, raw


def read_frame(path, t, w, h):
    for _t, raw in frames_at(path, [t], w, h):
        return np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    return None


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def dilate_px(width, base=3):
    """Scale a 1080p pixel dilation to the actual frame width."""
    return max(base, int(round(base * width / 1920.0)))


def build_roi(mask, fw, fh):
    """Pick the 512-wide band around the mark that LaMa will see.

    LaMa's input is fixed at 512x512 and the mark is a thin wide strip, so the
    band is stretched vertically to fill it. Padding with black instead makes
    the model treat the padding as real content and paint dark blobs into the
    hole, so stretching is the only safe option.

    Shared by the renderer and the preview so both see identical geometry;
    when they drift apart the preview shows artifacts the real render does not.
    """
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8)) if mask.ndim == 2 else mask
    bw = min(fw, 512)
    bh = int(round(bw * float(h) / float(max(1, w))))
    bh = max(64, min(bh, fh))
    y0 = max(0, min(fh - bh, y + h // 2 - bh // 2))
    x0 = max(0, min(fw - bw, x + w // 2 - bw // 2))
    return (x0, y0, x0 + bw, y0 + bh)


# --------------------------------------------------------------------------- #
# encode settings
# --------------------------------------------------------------------------- #
def crf_from_source(p):
    """Pick a CRF that matches how much real detail the source actually has.

    Over-crisp CRFs on an already-soft source only inflate the file. Under-crisp
    ones visibly smear the repaired patch and the rest of the frame.
    """
    w, h, fps = p["width"], p["height"], max(p["fps"], 1e-6)
    bpp = (p["bitrate"] / (w * h * fps)) if p["bitrate"] else 0.15
    if bpp >= 0.25:
        return 17, "high-bitrate source (%.3f bpp)" % bpp
    if bpp >= 0.15:
        return 18, "well-encoded source (%.3f bpp)" % bpp
    if bpp >= 0.08:
        return 19, "medium-bitrate source (%.3f bpp)" % bpp
    return 21, "already soft source (%.3f bpp)" % bpp


def encoder_args(encoder, crf):
    """Map a CRF-ish quality number onto whatever encoder the user's box has."""
    if encoder in (None, "auto"):
        encoder = "libx264"
    if encoder == "libx264":
        return ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p"]
    if encoder in ("h264_videotoolbox", "hevc_videotoolbox"):
        return ["-c:v", encoder, "-q:v", str(max(1, 100 - (crf - 10) * 4)),
                "-pix_fmt", "yuv420p", "-allow_sw", "1"]
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        return ["-c:v", encoder, "-rc", "vbr", "-cq", str(crf), "-preset", "p5",
                "-pix_fmt", "yuv420p"]
    if encoder in ("h264_qsv", "hevc_qsv"):
        return ["-c:v", encoder, "-global_quality", str(crf + 3), "-pix_fmt", "yuv420p"]
    if encoder in ("h264_amf", "hevc_amf"):
        return ["-c:v", encoder, "-qp_i", str(crf), "-qp_p", str(crf + 2), "-pix_fmt", "yuv420p"]
    raise SystemExit("unknown encoder %r (use libx264 or one of the hardware encoders)" % encoder)

class _Redirect308(urllib.request.HTTPRedirectHandler):
    """Python < 3.11 does not treat 308 as a redirect; GitHub's CDN sends one."""

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_302(req, fp, code, msg, headers)

    https_error_308 = http_error_308


def _download(url, dest, timeout=60, retries=3):
    """Stream a file to disk, retrying transient errors and 308 redirects."""
    import time
    last = None
    for attempt in range(retries):
        try:
            opener = urllib.request.build_opener(_Redirect308)
            opener.addheaders = [("User-Agent", "bilibili-wm-remover")]
            resp = opener.open(url, timeout=timeout)
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        print("\r  %5.1f%%  %.0f/%.0f MB" % (100.0 * got / total,
                                                             got / 1e6, total / 1e6),
                              end="", file=sys.stderr, flush=True)
            print("", file=sys.stderr)
            if got < 100_000_000:
                raise RuntimeError("only %d bytes received, expected ~205 MB" % got)
            return dest
        except Exception as e:
            last = e
            print("\n  attempt %d/%d failed: %s" % (attempt + 1, retries, e), file=sys.stderr)
            if os.path.exists(dest):
                os.remove(dest)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last))


def verify_sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    digest = h.hexdigest()
    if MODEL_SHA256 and digest != MODEL_SHA256:
        raise RuntimeError("checksum mismatch (got %s, want %s)"
                           % (digest[:12], MODEL_SHA256[:12]))
    return digest



# --------------------------------------------------------------------------- #
# torch device
# --------------------------------------------------------------------------- #
DEVICE_LABELS = {
    "mps": "Apple Silicon GPU (Metal)",
    "cuda": "NVIDIA / ROCm GPU",
    "xpu": "Intel GPU (oneAPI)",
    "dml": "DirectML GPU (any DX12 device)",
    "cpu": "CPU only",
}


def pick_device(prefer=None):
    """Return (device, label, all_available_devices)."""
    import torch

    opts = ["cpu"]
    try:
        if torch.backends.mps.is_available():
            opts.insert(0, "mps")
    except Exception:
        pass
    if torch.cuda.is_available():
        opts.insert(0, "cuda")
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            opts.insert(1, "xpu")
    except Exception:
        pass
    try:
        import torch_directml
        if torch_directml.is_available():
            opts.append("dml")
    except Exception:
        pass

    dev = prefer or opts[0]
    return dev, DEVICE_LABELS.get(dev, dev), opts


def torch_device(dev):
    import torch
    if dev == "dml":
        import torch_directml
        return torch_directml.device()
    return torch.device(dev)


def ensure_model(models_dir, url=None):
    os.makedirs(models_dir, exist_ok=True)
    target = os.path.join(models_dir, MODEL_NAME)
    if os.path.exists(target) and os.path.getsize(target) > 100_000_000:
        return target
    # A half-written .part from a previous run must not be resumed blindly:
    # its byte offsets would not line up with a different mirror.
    part = target + ".part"
    if os.path.exists(part):
        os.remove(part)
    for u in ([url] if url else MODEL_URLS):
        print("downloading LaMa weights (~205 MB)\n  from %s" % u, file=sys.stderr)
        try:
            _download(u, part)
            verify_sha(part)
            os.replace(part, target)
            print("  saved -> %s" % target, file=sys.stderr)
            return target
        except Exception as e:
            print("\n  failed: %s" % e, file=sys.stderr)
            if os.path.exists(part):
                os.remove(part)
    raise SystemExit(
        "could not download the LaMa model.\n"
        "  download it yourself and pass --weights /path/to/big-lama.pt\n"
        "  verified mirrors:\n    " + "\n    ".join(MODEL_URLS))
