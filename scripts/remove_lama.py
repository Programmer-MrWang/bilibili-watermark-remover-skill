#!/usr/bin/env python3
"""Method 2 - LaMa generative inpainting. The recommended method.

Rebuilds the masked pixels from surrounding context instead of smearing the
neighbours over them, which is why it survives detailed backgrounds (relief,
branches, fabric) where delogo leaves obvious streaks. Nothing outside the mask
is touched, so there is no seam.

Speed on a 1920x1080 / 30fps clip, per frame (the model always runs 512x512):
    Apple M5 (MPS)     ~0.23 s     ->  ~35 min for 5 minutes of video
    RTX 3060 (CUDA)    ~0.06 s     ->  ~10 min
    CPU only           ~4-6 s      ->  several hours
Everything else in the pipeline (decode, resize, encode) is ~3% of the cost.

Usage:
    python3 remove_lama.py in.mp4 out.mp4 --work wm_work --weights models/big-lama.pt
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


# The ROI is built by common.build_roi so the preview and the renderer always
# see identical geometry; when they drift apart, the preview shows artifacts
# the real render does not.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--work", default="wm_work")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None,
                    help="mps|cuda|xpu|dml|cpu (auto-detected when omitted)")
    ap.add_argument("--encoder", default="auto")
    ap.add_argument("--crf", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="render only the first N frames")
    C.add_fetch_args(ap)
    a = ap.parse_args()

    C.require_ffmpeg()
    a.src = C.resolve_source(a.src, height=a.height, cookies=a.cookies,
                             force=a.force_download, anonymous=a.anonymous)

    import torch
    torch.set_grad_enabled(False)

    p = C.probe(a.src)
    fw, fh = C.actual_frame_size(a.src, p["width"], p["height"])

    mask = cv2.imread(os.path.join(a.work, "mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        sys.exit("missing %s/mask.png - run detect_watermark.py first" % a.work)
    if mask.shape != (fh, fw):
        mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)

    rx0, ry0, rx1, ry1 = C.build_roi(mask, fw, fh)
    mroi = mask[ry0:ry1, rx0:rx1]
    if int(mroi.sum()) == 0:
        sys.exit("mask is empty inside the ROI - detection probably failed")

    dev, label, opts = C.pick_device(a.device)
    weights = a.weights or C.ensure_model(os.environ.get(
        "LAMA_MODELS", os.path.expanduser("~/.cache/bilibili-wm-remover")))
    model = torch.jit.load(weights, map_location="cpu").eval().to(C.torch_device(dev))

    crf, why = (a.crf, "user") if a.crf else C.crf_from_source(p)
    enc = C.encoder_args(a.encoder, crf)

    S = 512
    mup = cv2.resize((mroi > 0).astype(np.float32), (S, S), interpolation=cv2.INTER_NEAREST)
    m_t = torch.from_numpy(mup)[None, None].to(C.torch_device(dev))

    def fix(frame, out_buf):
        roi = frame[ry0:ry1, rx0:rx1]
        up = cv2.resize(roi, (S, S), interpolation=cv2.INTER_CUBIC)
        im = torch.from_numpy(up[:, :, ::-1].astype(np.float32) / 255.0)
        im = im.permute(2, 0, 1)[None].to(C.torch_device(dev))
        o = model(im, m_t)[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        back = cv2.resize((o.clip(0, 1) * 255).astype(np.uint8), (rx1 - rx0, ry1 - ry0),
                          interpolation=cv2.INTER_AREA)
        res = frame.copy()
        sel = mroi.astype(bool)
        res[ry0:ry1, rx0:rx1][sel] = back[sel]   # only masked pixels: no seam
        out_buf[:] = res

    audio_in = ["-i", a.src] if p["has_audio"] else []
    dec = subprocess.Popen([C.FFMPEG, "-loglevel", "error", "-i", a.src,
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    encp = subprocess.Popen(
        [C.FFMPEG, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (fw, fh),
         "-r", p["fps_str"], "-i", "-"] + audio_in +
        ["-map", "0:v"] + (["-map", "1:a:0?"] if p["has_audio"] else []) + enc +
        ["-c:a", "copy", "-shortest", a.dst], stdin=subprocess.PIPE)

    buf = fw * fh * 3
    frame = np.empty((fh, fw, 3), np.uint8)
    out = np.empty_like(frame)
    n, t0 = 0, time.time()
    total = a.limit or p["nb_frames"]
    try:
        while True:
            raw = dec.stdout.read(buf)
            if len(raw) < buf:
                break
            frame[:] = np.frombuffer(raw, np.uint8).reshape(fh, fw, 3)
            fix(frame, out)
            encp.stdin.write(out.tobytes())
            n += 1
            if n % 100 == 0 or n == total:
                el = time.time() - t0
                rate = n / el
                eta = (total - n) / rate if rate else 0
                print("\r  %d/%d frames  %.0f ms/frame  ETA %.0f min   "
                      % (n, total, 1000 / rate if rate else 0, eta / 60),
                      end="", file=sys.stderr, flush=True)
            if a.limit and n >= a.limit:
                break
    except BrokenPipeError:
        sys.stderr.write("\nencoder exited early\n")
    finally:
        print("", file=sys.stderr)
        dec.stdout.close()
        dec.kill()
        encp.stdin.close()
        encp.wait()

    if n == 0:
        sys.exit("no frames decoded - is %s a readable video?" % a.src)
    print("done -> %s (%d frames, %.1f min)" % (a.dst, n, (time.time() - t0) / 60),
          file=sys.stderr)


if __name__ == "__main__":
    main()
