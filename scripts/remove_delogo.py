#!/usr/bin/env python3
"""Method 3 - ffmpeg delogo. Fast, but only clean on low-detail backgrounds.

delogo interpolates the pixels inside the box from its border. When the
background has structure under the mark (relief carving, branches, fabric
folds, text), the interpolation cannot know what was there: it produces
horizontal smear and a visible checkerboard. That is fine for sky, walls,
darkness and defocused areas, so this script also offers a per-scene fallback:
it measures how much detail each shot has and can hand the hard shots to LaMa.

Usage:
    python3 remove_delogo.py in.mp4 out.mp4 --work wm_work [--encoder auto] [--crf N]
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


def detail_profile(src, pbox, samples=16):
    """Mean Laplacian energy of the ring around the mark, per sampled frame."""
    p = C.probe(src)
    fw, fh = C.actual_frame_size(src, p["width"], p["height"])
    x0, y0, x1, y1 = pbox
    ring_pad = max(12, (x1 - x0) // 12)
    vals = []
    for _t, raw, w, h in C.iter_sampled(src, samples):
        g = cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(h, w, 3), cv2.COLOR_BGR2GRAY)
        y0c, y1c = max(0, y0 - 3 * ring_pad), min(fh, y1 + 3 * ring_pad)
        x0c, x1c = max(0, x0 - 3 * ring_pad), min(fw, x1 + 3 * ring_pad)
        patch = g[y0c:y1c, x0c:x1c].astype(np.float32)
        vals.append(float(cv2.Laplacian(patch, cv2.CV_32F).std()))
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--work", default="wm_work")
    ap.add_argument("--encoder", default="auto")
    ap.add_argument("--crf", type=int, default=None)
    ap.add_argument("--pad", type=int, default=0,
                    help="extra pixels around the mask box (defaults to a scaled 3px)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    C.require_ffmpeg()
    p = C.probe(a.src)
    fw, fh = C.actual_frame_size(a.src, p["width"], p["height"])

    mask_path = os.path.join(a.work, "mask_tight.png")
    if not os.path.exists(mask_path):
        sys.exit("missing %s - run detect_watermark.py first" % mask_path)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m.shape != (fh, fw):
        m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)
    x, y, w, h = cv2.boundingRect((m > 127).astype(np.uint8))

    pad = a.pad or C.dilate_px(fw)
    bx = max(0, x - pad)
    by = max(0, y - pad)
    bw = min(fw, x + w + pad) - bx
    bh = min(fh, y + h + pad) - by
    # delogo requires the box to stay inside the frame
    bx, by = max(1, bx), max(1, by)
    bw, bh = min(fw - bx - 1, bw), min(fh - by - 1, bh)

    detail = detail_profile(a.src, [x, y, x + w, y + h])
    verdict = "smooth background" if float(detail.mean()) < 12 else \
              "DETAILED background - expect visible smear, prefer LaMa"

    crf, why = (a.crf, "user") if a.crf else C.crf_from_source(p)
    enc = C.encoder_args(a.encoder, crf)

    filt = "delogo=x=%d:y=%d:w=%d:h=%d" % (bx, by, bw, bh)
    cmd = [C.FFMPEG, "-y", "-loglevel", "error", "-stats", "-i", a.src, "-vf", filt] + enc
    if p["has_audio"]:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart"]
    if a.limit:
        cmd += ["-frames:v", str(a.limit)]
    cmd += [a.dst]

    print("delogo box: x=%d y=%d w=%d h=%d" % (bx, by, bw, bh), file=sys.stderr)
    print("background detail: mean Laplacian %.1f (p95 %.1f) -> %s"
          % (detail.mean(), np.percentile(detail, 95), verdict), file=sys.stderr)
    if float(detail.mean()) >= 12:
        print("  warning: this footage has structure under the mark; delogo will smear it.",
              file=sys.stderr)

    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit("ffmpeg failed (%d)" % r.returncode)

    with open(os.path.join(a.work, "delogo_report.json"), "w") as f:
        json.dump({"box": [bx, by, bw, bh], "detail_mean": float(detail.mean()),
                   "detail_p95": float(np.percentile(detail, 95)), "verdict": verdict,
                   "crf": crf, "crf_reason": why}, f, indent=2, ensure_ascii=False)
    print("done -> %s" % a.dst, file=sys.stderr)


if __name__ == "__main__":
    main()
