#!/usr/bin/env python3
"""Method 4 - alpha un-blend. Fast, exact in theory, limited by 8-bit data.

A Bilibili watermark is a semi-transparent white overlay, so a frame is
    obs = alpha*255 + (1-alpha)*bg
and the original picture can be recovered by inverting that per pixel:
    bg  = (obs - alpha*255) / (1 - alpha)

It is honest maths and it does work visually - but it divides the noise by
(1 - alpha). At alpha ~ 0.7 that is a 3.3x amplification of compression noise,
and where the underlying picture was near-white the signal is already clipped,
so no information is left to recover. Result: faint ghost outlines and coloured
fringes on the glyph edges. Treat this as a per-clip trick (flat / dark footage
works well) or as a pre-pass in front of LaMa - not as a general solution.

Usage:
    python3 remove_alpha.py in.mp4 out.mp4 --work wm_work
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--work", default="wm_work")
    ap.add_argument("--encoder", default="auto")
    ap.add_argument("--crf", type=int, default=None)
    ap.add_argument("--alpha-scale", type=float, default=1.0,
                    help="multiply the solved alpha (slightly >1 removes more, <1 is safer)")
    ap.add_argument("--edge-repair", type=float, default=0.0,
                    help="fraction of the mask width to inpaint at the rim (0 = off)")
    ap.add_argument("--limit", type=int, default=0)
    C.add_fetch_args(ap)
    a = ap.parse_args()

    C.require_ffmpeg()
    a.src = C.resolve_source(a.src, height=a.height, cookies=a.cookies,
                             force=a.force_download, anonymous=a.anonymous)
    p = C.probe(a.src)
    fw, fh = C.actual_frame_size(a.src, p["width"], p["height"])

    apath = os.path.join(a.work, "alpha.npy")
    if not os.path.exists(apath):
        sys.exit("missing %s - run detect_watermark.py first" % apath)
    alpha = np.load(apath)
    if alpha.shape != (fh, fw):
        alpha = cv2.resize(alpha, (fw, fh), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha * a.alpha_scale, 0.0, 0.95).astype(np.float32)

    tight = cv2.imread(os.path.join(a.work, "mask_tight.png"), cv2.IMREAD_GRAYSCALE)
    if tight is None:
        sys.exit("missing mask_tight.png - run detect_watermark.py first")
    if tight.shape != (fh, fw):
        tight = cv2.resize(tight, (fw, fh), interpolation=cv2.INTER_NEAREST)
    tight = (tight > 127).astype(np.uint8)
    rim = (cv2.dilate(tight, np.ones((3, 3), np.uint8)) ^
           cv2.erode(tight, np.ones((3, 3), np.uint8))).astype(np.uint8)

    crf, why = (a.crf, "user") if a.crf else C.crf_from_source(p)
    enc = C.encoder_args(a.encoder, crf)

    def fix(frame):
        out = frame.astype(np.float32)
        m = alpha > 0.02
        for c in range(3):
            ch = out[:, :, c]
            ch[m] = (ch[m] - alpha[m] * 255.0) / (1.0 - alpha[m] + 1e-3)
        out = np.clip(out, 0, 255).astype(np.uint8)
        if a.edge_repair > 0:
            out = cv2.inpaint(out, rim, max(1, int(a.edge_repair)), cv2.INPAINT_TELEA)
        return out

    dec = subprocess.Popen([C.FFMPEG, "-loglevel", "error", "-i", a.src,
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    encp = subprocess.Popen(
        [C.FFMPEG, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (fw, fh),
         "-r", p["fps_str"], "-i", "-"] + (["-i", a.src] if p["has_audio"] else []) +
        ["-map", "0:v"] + (["-map", "1:a:0?"] if p["has_audio"] else []) + enc +
        ["-c:a", "copy", "-shortest", a.dst], stdin=subprocess.PIPE)

    buf = fw * fh * 3
    n = 0
    try:
        while True:
            raw = dec.stdout.read(buf)
            if len(raw) < buf:
                break
            fr = np.frombuffer(raw, np.uint8).reshape(fh, fw, 3)
            encp.stdin.write(fix(fr).tobytes())
            n += 1
            if n % 500 == 0:
                print("  %d frames" % n, file=sys.stderr, flush=True)
            if a.limit and n >= a.limit:
                break
    finally:
        dec.stdout.close()
        dec.kill()
        encp.stdin.close()
        encp.wait()

    med = float(np.median(alpha[tight > 0])) if int(tight.sum()) else 0.0
    with open(os.path.join(a.work, "alpha_report.json"), "w") as f:
        json.dump({"frames": n, "alpha_median": round(med, 3),
                   "alpha_scale": a.alpha_scale, "edge_repair": a.edge_repair,
                   "crf": crf, "crf_reason": why,
                   "expectation": "ghost outlines appear where the background was bright; "
                                  "this is data loss, not a bug"}, f, indent=2, ensure_ascii=False)
    print("done -> %s (%d frames)" % (a.dst, n), file=sys.stderr)


if __name__ == "__main__":
    main()
