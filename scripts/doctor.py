#!/usr/bin/env python3
"""Check what this machine can actually do, and say how to fix what it cannot.

Run this first on a new machine, or whenever a link download behaves oddly.

    python3 scripts/doctor.py
    python3 scripts/doctor.py --from-browser chrome   # also export cookies

Nothing here downloads a video: it probes versions, devices and the cookie file,
so it is safe to run before a long render.
"""

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OK, WARN, BAD = "OK  ", "WARN", "MISS"


def line(status, label, detail="", fix=""):
    print("  [%s] %-16s %s" % (status, label, detail))
    if fix:
        for l in fix.splitlines():
            print("           " + l)


def check_python():
    v = sys.version.split()[0]
    ok = sys.version_info >= (3, 8)
    line(OK if ok else BAD, "python", v, "" if ok else "need 3.8+")


def check_modules():
    missing = []
    for mod in ("numpy", "cv2"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        line(BAD, "numpy/opencv", "missing: " + ", ".join(missing),
             "pip install -r requirements.txt")
        return False
    line(OK, "numpy/opencv", "importable")
    return True


def check_torch():
    try:
        import torch
    except Exception:
        line(WARN, "torch", "not installed - LaMa (method 2) unavailable",
             "install per https://pytorch.org/get-started/locally/\n"
             "then: python3 scripts/remove_lama.py --help\n"
             "methods 3 and 4 still work without it")
        return None
    try:
        dev, label, opts = C.pick_device()
        line(OK, "torch", "%s  ->  %s" % (torch.__version__, label),
             "" if dev != "cpu" else
             "no GPU backend found; CPU works but ~15x slower than MPS/CUDA")
        return dev
    except Exception as e:
        line(WARN, "torch", "installed but device probe failed: %s" % e)
        return "cpu"


def check_ffmpeg():
    for exe in (C.FFMPEG, C.FFPROBE):
        if not shutil.which(exe):
            line(BAD, exe, "not found",
                 "macOS:   brew install ffmpeg\n"
                 "Windows: winget install Gyan.FFmpeg\n"
                 "Linux:   sudo apt install ffmpeg")
            return False
    try:
        out = subprocess.run([C.FFMPEG, "-version"], capture_output=True, text=True,
                             timeout=20).stdout.splitlines()[0]
        line(OK, "ffmpeg", out.strip())
    except Exception as e:
        line(WARN, "ffmpeg", "found but failed to run: %s" % e)
    return True


def check_ytdlp():
    try:
        binary, version = C.find_ytdlp()
    except SystemExit as e:
        line(WARN, "yt-dlp", "not found - local files still work, links do not",
             str(e))
        return None
    line(OK, "yt-dlp", "%s (%s)" % (version, binary))
    return binary


def check_cookies():
    path, source = C.resolve_cookies_ex()
    st = C.cookie_status(path)
    if not path:
        line(WARN, "cookies", "not configured - links are limited to 720p",
             C.cookie_help())
        return None
    if not st["exists"]:
        line(BAD, "cookies", "%s (from %s) does not exist" % (path, source),
             "re-export it, or run: python3 scripts/doctor.py --from-browser chrome")
        return None
    bits = ["%d cookies" % st["count"], "SESSDATA" if st["has_sessdata"] else "NO SESSDATA",
            "netscape" if st["netscape"] else "NOT Netscape format"]
    status = OK if (st["has_sessdata"] and st["netscape"] and not st["expired"]) else WARN
    fix = ""
    if not st["netscape"]:
        fix = "this looks like JSON; export the Netscape cookies.txt format instead"
    elif not st["has_sessdata"]:
        fix = "no SESSDATA -> anonymous quality only. Log in to bilibili.com in that\nbrowser, then: python3 scripts/doctor.py --from-browser chrome"
    elif st["expired"]:
        fix = "SESSDATA expired; re-export the file"
    line(status, "cookies", "%s  (from %s)" % (", ".join(bits), source), fix)
    return path


def check_disk():
    d = C.media_dir()
    try:
        os.makedirs(d, exist_ok=True)
        free = shutil.disk_usage(d).free / 1e9
        status = OK if free > 5 else WARN
        line(status, "disk", "%.1f GB free at %s" % (free, d),
             "" if status == OK else "downloads need room; free some space or set BILI_WM_MEDIA_DIR")
    except Exception as e:
        line(WARN, "disk", "cannot use %s: %s" % (d, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-browser", default=None,
                    metavar="BROWSER",
                    help="export Bilibili cookies from a local browser "
                         "(chrome, edge, firefox, safari, ...) and exit")
    ap.add_argument("--cookies-out", default=None,
                    help="where to write the exported cookies (default: the skill config dir)")
    a = ap.parse_args()

    if a.from_browser:
        dest, n = C.export_cookies_from_browser(a.from_browser, a.cookies_out)
        print("exported %s cookies -> %s (mode 600)" % (n, dest), file=sys.stderr)
        st = C.cookie_status(dest)
        if not st["has_sessdata"]:
            print("WARNING: no SESSDATA in that browser - you are not logged in to "
                  "bilibili.com there, so this will still be anonymous quality.",
                  file=sys.stderr)
        print(dest)
        return 0

    print("bilibili-watermark-remover - environment check")
    print()
    ok_mods = check_modules()
    check_python()
    torch_dev = check_torch()
    ok_ff = check_ffmpeg()
    ytdlp = check_ytdlp()
    cookie = check_cookies()
    check_disk()

    print()
    ready_local = ok_mods and ok_ff
    ready_link = ready_local and ytdlp and cookie
    print("  local files : %s" % ("ready" if ready_local else "missing dependencies above"))
    if ready_local and not ytdlp:
        print("  links       : need yt-dlp (see above)")
    elif ready_link:
        print("  links       : ready, 1080p available")
    elif ytdlp:
        print("  links       : ready, but limited to 720p until cookies are set up")
    print("  LaMa        : %s" % ("ready on " + str(torch_dev) if torch_dev else
                                 "needs torch (methods 3/4 still work)"))
    return 0 if ready_local else 1


if __name__ == "__main__":
    sys.exit(main())
