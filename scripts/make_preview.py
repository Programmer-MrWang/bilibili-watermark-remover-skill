#!/usr/bin/env python3
"""Render the 4-up comparison still that the skill shows before any rendering.

One frame, four crops of the same region, labelled:
    original / method 2 LaMa / method 3 delogo / method 4 alpha

All four are computed from the SAME decoded frame in-process, so the picture
cannot drift between panels (rendering "one frame" through the movie scripts
would silently compare frame 0 against a frame at t=30s).

Costs a couple of seconds on a GPU box, tens of seconds on CPU. Rendering a
whole clip blind is rude; this lets the person choose with something concrete.

Usage:
    python3 make_preview.py in.mp4 --work wm_work -o preview.png [--at 30]
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# ASCII labels on purpose: cv2.putText cannot render CJK without a vector font
# bundled with the script, and a tofu-filled preview is worse than English.
LABELS = ["original", "2  LaMa (recommended)", "3  delogo", "4  alpha un-blend"]
PANEL_W, PANEL_H = 520, 220


def hardest_moment(src, p, mask):
    """Pick the sampled frame whose surroundings have the most detail.

    Watermark removal fails visibly on detailed backgrounds, so the honest
    preview shows the worst case rather than a flattering sky shot.
    """
    fw, fh = C.actual_frame_size(src, p["width"], p["height"])
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
    ring = (max(0, y - 40), min(fh, y + h + 40), max(0, x - 40), min(fw, x + w + 40))
    best_t, best = p["duration"] * 0.3, -1.0
    for i in range(16):
        t = p["duration"] * (i + 0.5) / 16
        fr = C.read_frame(src, t, fw, fh)
        if fr is None:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        d = float(cv2.Laplacian(g[ring[0]:ring[1], ring[2]:ring[3]], cv2.CV_32F).std())
        if d > best:
            best, best_t = d, t
    return best_t


def apply_alpha(frame, alpha, tight):
    out = frame.astype(np.float32)
    m = alpha > 0.02
    for c in range(3):
        ch = out[:, :, c]
        ch[m] = (ch[m] - alpha[m] * 255.0) / (1.0 - alpha[m] + 1e-3)
    out = np.clip(out, 0, 255).astype(np.uint8)
    rim = (cv2.dilate(tight, np.ones((3, 3), np.uint8)) ^
           cv2.erode(tight, np.ones((3, 3), np.uint8))).astype(np.uint8)
    return cv2.inpaint(out, rim, 2, cv2.INPAINT_TELEA)


def apply_delogo(src, t, box, fw, fh):
    """Run the real ffmpeg filter on that single frame, so the panel is honest."""
    r = subprocess.run([C.FFMPEG, "-v", "error", "-ss", "%.3f" % t, "-i", src,
                        "-vf", "delogo=x=%d:y=%d:w=%d:h=%d" % box,
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(fh, fw, 3).copy()


def apply_lama(frame, mask, weights, roi):
    import torch
    torch.set_grad_enabled(False)
    dev, _label, _opts = C.pick_device()
    model = torch.jit.load(weights, map_location="cpu").eval().to(C.torch_device(dev))
    rx0, ry0, rx1, ry1 = roi
    h, w = ry1 - ry0, rx1 - rx0
    mroi = mask[ry0:ry1, rx0:rx1]
    S = 512
    m_t = torch.from_numpy(
        (cv2.resize((mroi > 0).astype(np.float32), (S, S), interpolation=cv2.INTER_NEAREST))
    )[None, None].to(C.torch_device(dev))
    roi_img = frame[ry0:ry1, rx0:rx1]
    up = cv2.resize(roi_img, (S, S), interpolation=cv2.INTER_CUBIC)
    im = torch.from_numpy(up[:, :, ::-1].astype(np.float32) / 255.0)
    im = im.permute(2, 0, 1)[None].to(C.torch_device(dev))
    out = model(im, m_t)[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    back = cv2.resize((out.clip(0, 1) * 255).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_AREA)
    res = frame.copy()
    sel = mroi.astype(bool)
    res[ry0:ry1, rx0:rx1][sel] = back[sel]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--work", default="wm_work")
    ap.add_argument("-o", "--out", default="preview.png")
    ap.add_argument("--at", type=float, default=None, help="timestamp (seconds)")
    ap.add_argument("--weights", default=None)
    C.add_fetch_args(ap)
    a = ap.parse_args()

    C.require_ffmpeg()
    a.src = C.resolve_source(a.src, height=a.height, cookies=a.cookies,
                             force=a.force_download, anonymous=a.anonymous)
    p = C.probe(a.src)
    fw, fh = C.actual_frame_size(a.src, p["width"], p["height"])

    tight = cv2.imread(os.path.join(a.work, "mask_tight.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(a.work, "mask.png"), cv2.IMREAD_GRAYSCALE)
    alpha = np.load(os.path.join(a.work, "alpha.npy"))
    if tight is None or mask is None:
        sys.exit("missing masks - run detect_watermark.py first")
    for arr, name in ((tight, "mask_tight"), (mask, "mask")):
        if arr.shape != (fh, fw):
            raise SystemExit("%s is %s but the video decodes as %dx%d"
                             % (name, arr.shape, fw, fh))
    tight = (tight > 127).astype(np.uint8)
    mask = (mask > 127).astype(np.uint8)
    if alpha.shape != (fh, fw):
        alpha = cv2.resize(alpha, (fw, fh), interpolation=cv2.INTER_LINEAR)

    t = a.at if a.at is not None else hardest_moment(a.src, p, tight)
    frame = C.read_frame(a.src, t, fw, fh)
    if frame is None:
        sys.exit("could not decode a frame at %.1fs" % t)

    x, y, w, h = cv2.boundingRect(tight)
    pad = C.dilate_px(fw)
    box = (max(1, x - pad), max(1, y - pad),
           min(fw - max(1, x - pad) - 1, w + 2 * pad), min(fh - max(1, y - pad) - 1, h + 2 * pad))

    # Same ROI the renderer builds, so the two panels cannot drift apart
    roi = C.build_roi(mask, fw, fh)

    panels = [frame]
    weights = a.weights or os.path.join(os.environ.get(
        "LAMA_MODELS", os.path.expanduser("~/.cache/bilibili-wm-remover")), "big-lama.pt")
    try:
        import torch  # noqa: F401
        if not os.path.exists(weights):
            weights = C.ensure_model(os.path.dirname(weights))
        panels.append(apply_lama(frame, mask, weights, roi))
    except Exception as e:
        panels.append(None)
        print("  LaMa panel skipped: %s" % repr(e)[:160], file=sys.stderr)

    panels.append(apply_delogo(a.src, t, box, fw, fh))
    panels.append(apply_alpha(frame, alpha, tight))

    crop = (max(0, x - 30), max(0, y - 30),
            min(fw, x + w + 30), min(fh, y + h + 30))
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    scale = min(PANEL_W / float(cw), PANEL_H / float(ch))
    pw, ph = int(cw * scale), int(ch * scale)

    tiles = []
    for fr in panels:
        if fr is None:
            tile = np.full((PANEL_H, PANEL_W, 3), 24, np.uint8)
            cv2.putText(tile, "unavailable", (150, 118), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (90, 90, 90), 2)
        else:
            tile = cv2.resize(fr[crop[1]:crop[3], crop[0]:crop[2]], (pw, ph),
                              interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 0, PANEL_H - ph, 0, PANEL_W - pw,
                                      cv2.BORDER_CONSTANT, value=(20, 20, 20))
        tiles.append(tile)

    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    for i, label in enumerate(LABELS):
        cx, cy = (i % 2) * PANEL_W, (i // 2) * PANEL_H
        cv2.rectangle(grid, (cx, cy), (cx + 200, cy + 32), (0, 0, 0), -1)
        cv2.putText(grid, label, (cx + 8, cy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2)
    cv2.putText(grid, "t=%.1fs  %dx%d" % (t, fw, fh), (8, grid.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    cv2.imwrite(a.out, grid)
    print("preview -> %s  (frame at %.2fs, %dx%d)" % (a.out, t, fw, fh), file=sys.stderr)


if __name__ == "__main__":
    main()
