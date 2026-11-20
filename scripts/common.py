"""Shared helpers for the Bilibili watermark remover.

Deliberately dependency-light and cross-platform:
  - probing / decoding go through ffmpeg only
  - device selection probes mps / cuda / xpu / directml / cpu in that order
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
import cv2

def _ffmpeg_default():
    """Honour an explicit path before falling back to a PATH lookup.

    DSH_FFMPEG_PATH is what the DSH desktop app sets: it ships ffmpeg under
    $DSH_HOME/bin, which is often not on PATH when a script is launched from a
    GUI. ffprobe normally sits next to ffmpeg.
    """
    for var in ("FFMPEG_BIN", "DSH_FFMPEG_PATH"):
        v = os.environ.get(var)
        if v:
            return v
    return "ffmpeg"


def _ffprobe_for(ffmpeg):
    v = os.environ.get("FFPROBE_BIN")
    if v:
        return v
    d = os.path.dirname(ffmpeg)
    if d:
        cand = os.path.join(d, "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.exists(cand):
            return cand
    return "ffprobe"


FFMPEG = _ffmpeg_default()
FFPROBE = _ffprobe_for(FFMPEG)

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
    """Yield (`t`, raw_bgr_bytes, w, h) for ~`want` evenly spaced frames.

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


# --------------------------------------------------------------------------- #
# input resolution: local file or a Bilibili / yt-dlp supported link
# --------------------------------------------------------------------------- #
URL_RE = re.compile(r"^(?:https?|ftp)://", re.I)

YTDLP_ENV = "DSH_YTDLP_PATH"        # same variable the DSH video plugin honours
COOKIES_ENV = "DSH_VIDEO_COOKIES"   # ditto
DSH_VIDEO_JSON = os.path.join(os.path.expanduser("~"), ".dsh", "video.json")
MEDIA_ENV = "BILI_WM_MEDIA_DIR"

_YTDLP_HINT = (
    "yt-dlp not found, and it is what turns a link into a file.\n"
    "  macOS:   brew install yt-dlp\n"
    "  Windows: winget install yt-dlp     (or: pip install --user -U yt-dlp)\n"
    "  Linux:   pip install --user -U yt-dlp   (or your distro package)\n"
    "  then either put it on PATH or set %s=/path/to/yt-dlp" % YTDLP_ENV
)


def is_url(src):
    return bool(URL_RE.match(str(src)))


def _ytdlp_candidates(env=None):
    env = env or os.environ
    out = []
    if env.get(YTDLP_ENV):
        out.append(env[YTDLP_ENV])
    which = shutil.which("yt-dlp", path=env.get("PATH") or os.defpath)
    if which:
        out.append(which)
    home = os.path.expanduser("~")
    exe = "yt-dlp.exe" if os.name == "nt" else "yt-dlp"
    dirs = [os.path.join(home, ".local", "bin")]
    if sys.platform == "darwin":
        dirs += ["/opt/homebrew/bin", "/usr/local/bin"]
    if os.name == "nt" and env.get("APPDATA"):
        dirs.append(os.path.join(env["APPDATA"], "Python"))
    for d in dirs:
        out.append(os.path.join(d, exe))
        try:
            for child in os.listdir(d):
                out.append(os.path.join(d, child, "Scripts", exe))
        except OSError:
            pass
    seen = set()
    return [p for p in out if p and not (p in seen or seen.add(p))]


def find_ytdlp(env=None):
    """Return (binary, version). A file that exists but cannot run does not count.

    Existence alone is a trap: a half-copied binary or a wrong-architecture build
    fails at spawn time, which then looks like "yt-dlp is broken" much later.
    """
    tried = []
    for cand in _ytdlp_candidates(env):
        if not os.path.exists(cand):
            continue
        tried.append(cand)
        try:
            r = subprocess.run([cand, "--version"], capture_output=True, text=True, timeout=30)
        except Exception:
            continue
        version = (r.stdout or "").strip().splitlines()
        if r.returncode == 0 and version:
            return cand, version[0].strip()
    err = _YTDLP_HINT
    if tried:
        err += "\n  tried: " + ", ".join(tried[:4])
    raise SystemExit(err)


def resolve_cookies(explicit=None, env=None):
    """explicit arg > DSH_VIDEO_COOKIES > ~/.dsh/video.json "cookies" > none.

    The DSH video plugin uses exactly these three sources, so an installed DSH
    with a logged-in cookies.txt needs no extra setup here.
    """
    env = env or os.environ
    for cand in (explicit, env.get(COOKIES_ENV)):
        if cand:
            if not os.path.exists(cand):
                raise SystemExit("cookies file not found: %s" % cand)
            return cand
    try:
        with open(DSH_VIDEO_JSON, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        p = cfg.get("cookies") if isinstance(cfg, dict) else None
        if p and os.path.exists(p):
            return p
    except (OSError, ValueError):
        pass
    return None


def skill_config_dir(env=None):
    """Where this skill keeps its own settings (not DSH's)."""
    env = env or os.environ
    if os.name == "nt":
        base = env.get("APPDATA") or os.path.expanduser("~")
    else:
        base = env.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "bilibili-wm-remover")


def skill_cookies_path(env=None):
    return os.path.join(skill_config_dir(env), "cookies.txt")


def cookie_status(path):
    """Is this a usable Netscape cookie file, and is it signed in?"""
    info = {"path": path, "exists": False, "netscape": False, "has_sessdata": False,
            "expired": False, "count": 0}
    if not path or not os.path.exists(path):
        return info
    info["exists"] = True
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return info
    info["netscape"] = text.lstrip().startswith("# Netscape")
    now = time.time()
    for line in text.splitlines():
        if not line or line.startswith("#") or "\t" not in line:
            continue
        f5 = line.split("\t")
        if len(f5) < 7:
            continue
        info["count"] += 1
        if f5[5] == "SESSDATA":
            info["has_sessdata"] = True
            try:
                # yt-dlp writes microseconds for session cookies; treat huge
                # values as "session, does not expire".
                exp = float(f5[4]) / 1e6
                if exp < now:
                    info["expired"] = True
            except ValueError:
                pass
    return info


def resolve_cookies_ex(explicit=None, env=None, anonymous=False):
    """Return (path_or_None, source_label).

    Order: --cookies > $DSH_VIDEO_COOKIES > ~/.dsh/video.json > this skill's own
    config dir. The DSH entries come first so a machine already logged in for the
    DSH video tools needs no second setup; the skill's own file is the fallback
    for people who do not use DSH at all.
    """
    env = env or os.environ
    if anonymous:
        return None, "disabled by --anonymous"
    for cand, label in ((explicit, "--cookies"),
                        (env.get(COOKIES_ENV), "$" + COOKIES_ENV)):
        if cand:
            if not os.path.exists(cand):
                raise SystemExit("cookies file not found: %s" % cand)
            return cand, label
    try:
        with open(DSH_VIDEO_JSON, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        p = cfg.get("cookies") if isinstance(cfg, dict) else None
        if p and os.path.exists(p):
            return p, DSH_VIDEO_JSON
    except (OSError, ValueError):
        pass
    own = skill_cookies_path(env)
    if os.path.exists(own):
        return own, own
    return None, "not configured"


def resolve_cookies(explicit=None, env=None):
    return resolve_cookies_ex(explicit, env)[0]


def cookie_help(height=1080):
    """The message a user sees when 1080p is wanted but nobody is logged in."""
    own = skill_cookies_path()
    return (
        "  You asked for %dp. Bilibili caps anonymous requests at 720p for most\n"
        "  uploads (a few allow 1080p), so going higher needs a signed-in session.\n"
        "  Three ways to get one:\n"
        "\n"
        "  1) If you already use DSH's video tools, they are set up: nothing to do.\n"
        "     This skill reads $DSH_VIDEO_COOKIES and ~/.dsh/video.json first.\n"
        "\n"
        "  2) Let this skill pull them from your browser (needs yt-dlp):\n"
        "       python3 scripts/doctor.py --from-browser chrome      # or edge, firefox, safari\n"
        "     It writes %s (mode 600).\n"
        "\n"
        "  3) Export cookies.txt with a browser extension (Netscape format, not JSON)\n"
        "     and point the skill at it:\n"
        "       --cookies /path/to/cookies.txt\n"
        "     or  export %s=/path/to/cookies.txt\n"
        "\n"
        "  Either way the file must come from a logged-in bilibili.com session and\n"
        "  contain SESSDATA. It expires eventually; re-export when it does.\n"
        "  Run 'python3 scripts/doctor.py' to check the current state."
        % (height, own, COOKIES_ENV))


def export_cookies_from_browser(browser, dest=None, env=None):
    """Dump Bilibili cookies from a local browser into a Netscape file.

    Uses yt-dlp's own cookie reader, so it understands the Keychain / DPAPI
    storage on each OS without a browser extension. Chromium 127+ may refuse
    (App-Bound Encryption) - that is reported, not hidden.
    """
    dest = dest or skill_cookies_path(env)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    ytdlp, _version = find_ytdlp(env)

    code = (
        "import http.cookiejar, sys\n"
        "from yt_dlp.cookies import extract_cookies_from_browser as x\n"
        "b, out = sys.argv[1], sys.argv[2]\n"
        "jar = x(b)\n"
        "keep = ('bilibili', 'bilivideo', 'biliapi')\n"
        "w = http.cookiejar.MozillaCookieJar(out)\n"
        "n = 0\n"
        "for c in jar:\n"
        "    if any(k in (c.domain or '') for k in keep):\n"
        "        w.set_cookie(c); n += 1\n"
        "if not n:\n"
        "    sys.exit('no bilibili cookies found in ' + b + '; log in there first')\n"
        "w.save(ignore_discard=True, ignore_expires=True)\n"
        "print(n)\n"
    )
    # yt-dlp's own interpreter first (pip installs expose yt_dlp to it), then ours.
    interpreters = []
    real = os.path.realpath(ytdlp)
    for up in (os.path.dirname(os.path.dirname(real)),):
        cand = os.path.join(up, "bin", "python")
        if os.path.exists(cand):
            interpreters.append(cand)
    interpreters.append(sys.executable)

    last = None
    for py in interpreters:
        r = subprocess.run([py, "-c", code, browser, dest],
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            try:
                os.chmod(dest, 0o600)
            except OSError:
                pass
            return dest, (r.stdout or "").strip()
        last = (r.stderr or "").strip() or (r.stdout or "").strip()
        if "ModuleNotFoundError" not in last:
            break
    # Report the exception line, not the top of the traceback: the reason is
    # always at the bottom and the head is just call frames.
    tail = [l.strip() for l in (last or "").splitlines() if l.strip()]
    detail = tail[-1] if tail else "no output"
    low = (last or "").lower()
    if "app-bound" in low or "decrypt" in low or "dpapi" in low or "permission" in low:
        raise SystemExit(
            "could not read %s's cookie store.\n"
            "  %s\n"
            "  Chromium 127+ encrypts cookies (App-Bound / DPAPI) in a way yt-dlp\n"
            "  cannot always decrypt, and macOS may also be blocking the Keychain\n"
            "  prompt. Workarounds, in order of success rate:\n"
            "    a) try another browser:  --from-browser safari   (or firefox)\n"
            "    b) export cookies.txt with a browser extension (Netscape format)\n"
            "    c) if you use DSH, its video tools may already have a working file"
            % (browser, detail))
    raise SystemExit("cookie export failed for %s:\n  %s" % (browser, detail))


def media_dir(env=None):
    env = env or os.environ
    return env.get(MEDIA_ENV) or os.path.join(
        os.path.expanduser("~"), ".cache", "bilibili-wm-remover", "media")


def _explain_ytdlp(stderr, code):
    t = str(stderr or "")
    if re.search(r"may be deleted|geo-restricted|稿件不可见|不存在", t, re.I):
        return "the video is gone or region-locked (not a tool failure)"
    if re.search(r"Sign in to confirm|login required|需要登录|Please log in", t, re.I):
        return ("this video needs a signed-in session (1080P+, member-only or paywalled); "
                "point --cookies at a cookies.txt, or set %s" % COOKIES_ENV)
    if re.search(r"HTTP Error 412", t):
        return "rate limited (HTTP 412); wait a moment and retry"
    if re.search(r"HTTP Error 429", t):
        return "too many requests (HTTP 429); wait and retry"
    if re.search(r"HTTP Error 567", t):
        return ("the site refused the request (HTTP 567) - usually needs a signed-in session; "
                "supply cookies, or download the file yourself and pass a local path")
    if re.search(r"Unsupported URL", t):
        return "yt-dlp does not support this URL"
    tail = " | ".join([l for l in t.strip().splitlines() if l.strip()][-2:])
    return "yt-dlp exited %s%s" % (code, (": " + tail) if tail else "")


def fetch_video(url, height=1080, cookies=None, force=False, quiet=False, env=None,
                anonymous=False):
    """Download url (best <= height, audio included) and return metadata.

    Returns {"path","title","duration","height","url","cached","auth"}.
    """
    env = env or os.environ
    ytdlp, version = find_ytdlp(env)
    cookie_path, cookie_src = resolve_cookies_ex(cookies, env, anonymous=anonymous)

    key = hashlib.sha1(("%s|%s" % (url, height)).encode("utf-8")).hexdigest()[:20]
    outdir = os.path.join(media_dir(env), key)
    meta_path = os.path.join(outdir, "meta.json")
    os.makedirs(outdir, exist_ok=True)

    if not force:
        for name in sorted(os.listdir(outdir)):
            if name.startswith("source.") and not name.endswith(".part"):
                p = os.path.join(outdir, name)
                if os.path.getsize(p) > 1024:
                    meta = {}
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except (OSError, ValueError):
                        pass
                    return {"path": p, "title": meta.get("title"),
                            "duration": meta.get("duration"),
                            "height": meta.get("height", height),
                            "url": url, "cached": True,
                            "auth": meta.get("auth", cookie_src)}

    common_flags = ["--ignore-config", "--no-warnings", "--no-playlist"]
    base = [ytdlp] + common_flags
    if cookie_path:
        base += ["--cookies", cookie_path]

    # Probe first: it fails fast with a readable reason, and gives us the title.
    title = duration = None
    if not quiet:
        print("  probing %s" % url, file=sys.stderr)
    # NB: do not name this "probe" - that shadows the module-level probe()
    # used further down to read the height of the downloaded file.
    meta_run = subprocess.run(base + ["-J", url], capture_output=True, text=True,
                              timeout=180)
    if meta_run.returncode != 0:
        raise SystemExit("cannot read %s\n  %s"
                         % (url, _explain_ytdlp(meta_run.stderr, meta_run.returncode)))
    try:
        info = json.loads(meta_run.stdout)
        if info.get("_type") == "playlist":
            entries = [e for e in (info.get("entries") or []) if e]
            if not entries:
                raise SystemExit("no playable entry at %s (multi-part link? add ?p=N)" % url)
            info = entries[0]
        title = info.get("title")
        duration = info.get("duration")
        heights = sorted({int(f["height"]) for f in (info.get("formats") or [])
                          if f.get("vcodec") not in (None, "none")
                          and isinstance(f.get("height"), int) and f["height"] > 0})
        best = heights[-1] if heights else None
        if best and best < height and not quiet:
            # Not a failure: say what is actually available and why.
            if not cookie_path:
                print("\n  only %dp is available without signing in." % best, file=sys.stderr)
                print(cookie_help(height), file=sys.stderr)
            else:
                print("\n  note: this upload tops out at %dp (signed-in as per %s)"
                      % (best, cookie_src), file=sys.stderr)
    except SystemExit:
        raise
    except ValueError:
        pass

    # Prefer a progressive stream (one file, no muxing); fall back to DASH,
    # which is what Bilibili actually serves above 720p - that needs ffmpeg.
    sel = ("bv*[height<=%d]+ba/b[height<=%d][acodec!=none]/b[height<=%d]/bv*+ba/b"
           % (height, height, height))
    args = base + [
        "-f", sel,
        "-S", "res:%d" % height,
        "--merge-output-format", "mp4",
        "--continue",
        "--retries", "3", "--fragment-retries", "5",
        "--no-progress",
        "-o", os.path.join(outdir, "source.%(ext)s"),
        url,
    ]
    if not quiet:
        print("  downloading (<=%dp) via yt-dlp %s%s"
              % (height, version, "" if cookie_path else "  [anonymous: <=720p]"),
              file=sys.stderr)

    dl = subprocess.run(args, capture_output=True, text=True)
    video = None
    for name in sorted(os.listdir(outdir)):
        if name.startswith("source.") and not name.endswith(".part"):
            p = os.path.join(outdir, name)
            if os.path.getsize(p) > 1024:
                video = p
                break

    # A non-zero exit can still leave a usable file (post-processing failures are
    # common); the file is what matters, not the exit code.
    if video is None:
        raise SystemExit("download failed\n  %s" % _explain_ytdlp(dl.stderr, dl.returncode))

    # Trust the file, not the probe: Bilibili will happily list a format it then
    # refuses to serve, and the probe runs before any rate limiting kicks in.
    actual = None
    try:
        actual = probe(video)["height"]
    except Exception:
        pass
    if actual and actual < height and not quiet:
        if not cookie_path:
            print("\n  got %dp, not %dp - anonymous quality is capped by the site."
                  % (actual, height), file=sys.stderr)
            print(cookie_help(height), file=sys.stderr)
        elif actual < height:
            print("\n  got %dp: this upload tops out there (signed in per %s)"
                  % (actual, cookie_src), file=sys.stderr)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"url": url, "title": title, "duration": duration,
                   "height": actual or height, "requested": height,
                   "auth": cookie_src, "ytdlp": version}, f,
                  ensure_ascii=False, indent=2)

    if not quiet:
        print("  saved -> %s (%dp)" % (video, actual) if actual else
              "  saved -> %s" % video, file=sys.stderr)
    return {"path": video, "title": title, "duration": duration,
            "height": actual or height, "url": url, "cached": False,
            "auth": cookie_src}


def add_fetch_args(ap):
    """Flags every entry point shares for link inputs."""
    ap.add_argument("--height", type=int, default=1080,
                    help="max height when the input is a link (default 1080)")
    ap.add_argument("--cookies", default=None,
                    help="cookies.txt for login-gated links (default: DSH_VIDEO_COOKIES "
                         "or the cookies field in ~/.dsh/video.json)")
    ap.add_argument("--force-download", action="store_true",
                    help="re-download even if a cached copy exists")
    ap.add_argument("--anonymous", action="store_true",
                    help="ignore cookies and fetch anonymously (<=720p); useful to "
                         "see what works before setting up a signed-in session")


def resolve_source(src, height=1080, cookies=None, force=False, quiet=False,
                   anonymous=False):
    """Accept a local path or a link; always return a readable local path."""
    if not is_url(src):
        if not os.path.exists(src):
            raise SystemExit("input not found: %s" % src)
        return src
    info = fetch_video(src, height=height, cookies=cookies, force=force, quiet=quiet,
                       anonymous=anonymous)
    if info.get("title") and not quiet:
        print("  %s" % info["title"], file=sys.stderr)
    return info["path"]


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
