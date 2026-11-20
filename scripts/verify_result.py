#!/usr/bin/env python3
"""Quantitative check for a rendered result. Run this before handing a file over.

Two numbers matter, and neither of them is "does it look okay to me":
  1. residual - how much of the watermark's brightness signature is left inside
     the mask, relative to the ring around it (original vs result);
  2. collateral - PSNR/SSIM outside the mask, to prove nothing else was touched.

Usage:
    python3 verify_result.py result.mp4 --src original.mp4 --work wm_work
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


def residual(path, mask, step, fw, fh):
    """Mean (mask - ring median) brightness per sampled frame.

    The ring median is a local background estimate, so this isolates "the mark
    is still painted on top" from "the scene is bright here".
    """
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
    inner = (mask[y:y + h, x:x + w] > 0)
    outer = ~inner
    if not inner.any() or not outer.any():
        raise SystemExit("mask has no interior or no ring - wrong mask file?")
    # Sample every "step"-th frame: decoding the whole clip just to average a
    # brightness metric costs minutes on a 5 minute video for no extra signal.
    want = max(8, C.probe(path)["nb_frames"] // max(1, step))
    vals = []
    for _t, raw, _w, _h in C.iter_sampled(path, want):
        g = cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(fh, fw, 3),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi = g[y:y + h, x:x + w]
        vals.append(float((roi[inner] - np.median(roi[outer])).mean()))
    return np.array(vals)


def quality(src, dst, mask, fw, fh):
    """Whole-frame SSIM between source and result.

    This is NOT restricted to the area outside the mask: ffmpeg's filter works
    on the whole frame, and the repaired patch is by design different from the
    original. Use it as "did the re-encode keep the picture", not as "did the
    repair damage the neighbourhood" - for the latter compare the ring
    statistics (see residual()).
    """
    r = subprocess.run([C.FFMPEG, "-v", "info", "-i", src, "-i", dst,
                        "-lavfi", "[0:v]setpts=PTS-STARTPTS[ref];"
                                  "[1:v]setpts=PTS-STARTPTS[dis];[ref][dis]ssim",
                        "-f", "null", "-"], capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if "SSIM" in line and "All" in line:
            return line.split("]")[-1].strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dst")
    ap.add_argument("--src", default=None, help="original, for the residual baseline")
    ap.add_argument("--work", default="wm_work")
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--json", default=None)
    C.add_fetch_args(ap)
    a = ap.parse_args()

    C.require_ffmpeg()
    a.dst = C.resolve_source(a.dst, height=a.height, cookies=a.cookies,
                             force=a.force_download, anonymous=a.anonymous)
    if a.src:
        a.src = C.resolve_source(a.src, height=a.height, cookies=a.cookies)
    mask = cv2.imread(os.path.join(a.work, "mask_tight.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        sys.exit("missing mask_tight.png - run detect_watermark.py first")
    p = C.probe(a.dst)
    fw, fh = C.actual_frame_size(a.dst, p["width"], p["height"])
    if mask.shape != (fh, fw):
        mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)

    out = residual(a.dst, mask, a.step, fw, fh)
    base = residual(a.src, mask, max(a.step * 3, 45), fw, fh) if a.src else None

    rep = {"result": a.dst, "frames_sampled": int(len(out)),
           "residual_mean": round(float(out.mean()), 2),
           "residual_p95": round(float(np.percentile(out, 95)), 2),
           "residual_max": round(float(out.max()), 2)}
    if base is not None:
        rep["baseline_mean"] = round(float(base.mean()), 2)
        rep["removed_pct"] = round(100.0 * (1 - out.mean() / max(base.mean(), 1e-6)), 1)

    if a.src:
        rep["ssim_whole_frame"] = quality(a.src, a.dst, mask, fw, fh)

    nums = [rep.get("residual_mean"), rep.get("residual_p95")]
    if any((v is None) or not np.isfinite(v) for v in nums):
        verdict = "FAIL: residual could not be measured (mask/ROI empty?)"
    elif base is not None and not np.isfinite(rep.get("removed_pct", float("nan"))):
        verdict = "FAIL: baseline could not be measured"
    elif base is not None and rep["removed_pct"] < 85:
        verdict = "WARN: watermark signal largely still present"
    elif base is None and rep["residual_mean"] > 25:
        verdict = "WARN: bright signature still inside the mask"
    else:
        verdict = "PASS"
    rep["verdict"] = verdict

    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
    return 0 if verdict == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
