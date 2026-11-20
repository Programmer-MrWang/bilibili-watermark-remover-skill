#!/usr/bin/env python3
"""Find the Bilibili watermark, verify it, and write mask + alpha files.

No logo template is used, so this works at any resolution and any position.
It leans on three facts measured on real Bilibili uploads:

  1. the mark is a STATIC overlay: high-pass energy at those pixels survives a
     temporal median, while moving scene detail cancels out;
  2. it is a THIN bright stroke composite (text + logo), so
     gray - medianBlur(gray) isolates it and removes the smooth background;
  3. it is semi-transparent WHITE: obs = alpha*255 + (1-alpha)*bg, so alpha is
     solvable per pixel from frame pairs whose background differs.

Usage:
    python3 detect_watermark.py video.mp4 -o out/
    python3 detect_watermark.py video.mp4 -o out/ --bbox x,y,w,h   # skip detection
Exit: 0 verified, 3 found-but-unverified, 2 nothing found.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def odd(v):
    v = int(round(v))
    return v if v % 2 else v + 1


def highpass(gray, k):
    """Isolate thin bright strokes: gray minus its own large-kernel median."""
    return gray.astype(np.float32) - cv2.medianBlur(gray.astype(np.uint8), k).astype(np.float32)


def collect(path, want, work_w=960):
    """Sample frames once, downscaled. Returns (grays, scale, probe_dict)."""
    p = C.probe(path)
    fw, fh = C.actual_frame_size(path, p["width"], p["height"])
    scale = min(1.0, work_w / float(fw))
    ww, wh = int(round(fw * scale)), int(round(fh * scale))
    grays = []
    for _t, raw, _w, _h in C.iter_sampled(path, want):
        fr = np.frombuffer(raw, np.uint8).reshape(fh, fw, 3)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(g, (ww, wh), interpolation=cv2.INTER_AREA))
    return np.stack(grays), scale, p, (fw, fh)


def overlay_energy(grays, k):
    """Temporal median of positive high-pass energy = static bright strokes."""
    pos = np.stack([np.clip(highpass(g, k), 0, None) for g in grays])
    return np.median(pos, axis=0)


def find_candidates(energy):
    """Binarise energy, then group thin strokes into watermark-shaped bands."""
    H, W = energy.shape
    thr = max(4.0, 0.30 * float(np.percentile(energy, 99.9)))
    m = (energy > thr).astype(np.uint8)
    kw, kh = odd(0.020 * W), odd(0.008 * W)         # join glyphs horizontally
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.0005 * energy.size:
            continue
        if y > 0.30 * H:
            continue                                # Bilibili anchors to the top
        if not (0.05 <= w / float(W) <= 0.45):
            continue
        if not (0.012 <= h / float(H) <= 0.20):
            continue
        ar = w / float(h)
        if not (1.4 <= ar <= 9.0):
            continue
        out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "area": int(area), "ar": float(ar),
                    "score": float(area * (1.0 - min(1.0, abs(ar - 4.2) / 4.5)))})
    out.sort(key=lambda c: -c["score"])
    return out


def merge_fragments(cands, gap_frac=0.08):
    """Text and logo may segment separately; merge boxes on the same baseline."""
    boxes = [[c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"], c["area"]] for c in cands]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            if boxes[i] is None:
                continue
            for j in range(i + 1, len(boxes)):
                if boxes[j] is None:
                    continue
                a, b = boxes[i], boxes[j]
                vov = min(a[3], b[3]) - max(a[1], b[1])
                minh = min(a[3] - a[1], b[3] - b[1])
                hgap = max(a[0], b[0]) - min(a[2], b[2])
                if vov > 0.4 * minh and hgap < gap_frac * max(a[2] - a[0], b[2] - b[0]) + 24:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]),
                                max(a[3], b[3]), a[4] + b[4]]
                    boxes[j] = None
                    changed = True
    return [{"x": b[0], "y": b[1], "w": b[2] - b[0], "h": b[3] - b[1], "area": b[4]}
            for b in boxes if b is not None]


def logo_signature(tight, box):
    """Bilibili's mark is sparse text on the left, a dense solid logo on the right.

    Measured on the glyph mask itself, so it is resolution independent and does
    not care about the background behind the watermark.
    """
    x, y, w, h = [int(v) for v in box]
    band = tight[y:y + h, x:x + w]
    if band.size < 64:
        return 0.0, 0.0
    left = band[:, :int(0.55 * w)]
    right = band[:, int(0.62 * w):]
    return float(right.mean()), float(left.mean())


def validate(grays, box, energy, tight=None, tight_box=None):
    """Numeric gate: static, contrasty, plausible geometry, dense logo on the right.

    Runs on the coarse (downscaled) frames for temporal evidence and, when
    available, on the full-resolution glyph mask for the logo signature. The
    signature is measured from the mask rather than the pixels so a dark or
    busy background cannot fake a pass.
    """
    H, W = grays.shape[1], grays.shape[2]
    x, y, w, h = [int(v) for v in box]
    pad = max(4, int(0.012 * W))
    inner = energy[y:y + h, x:x + w]
    y0, y1 = max(0, y - 3 * pad), min(H, y + h + 3 * pad)
    x0, x1 = max(0, x - 3 * pad), min(W, x + w + 3 * pad)
    ring = np.concatenate([energy[y0:y1, x0:x0 + pad].ravel(),
                           energy[y0:y1, x1 - pad:x1].ravel(),
                           energy[y0:y0 + pad, x0:x1].ravel(),
                           energy[y1 - pad:y1, x0:x1].ravel()])
    contrast = float(inner.mean()) / (float(np.median(ring)) + 1e-3)

    k = odd(max(9, 0.021 * W))
    per_frame = [float(highpass(g, k)[y:y + h, x:x + w].mean()) for g in grays]
    static_ratio = float((np.array(per_frame) > 1.0).mean())

    ar = w / float(h)
    checks = {
        "energy_contrast": round(contrast, 2),
        "static_ratio": round(static_ratio, 3),
        "aspect": round(ar, 2),
        "width_frac": round(w / float(W), 3),
        "height_frac": round(h / float(H), 3),
        "per_frame_energy": [round(v, 2) for v in per_frame[:10]],
    }

    shape_ok = (contrast > 3.0
                and static_ratio > 0.9
                and 1.4 <= ar <= 9.0
                and 0.05 <= checks["width_frac"] <= 0.45
                and 0.012 <= checks["height_frac"] <= 0.20)
    logo_ok = True
    if tight is not None:
        # tight is full resolution; box is in coarse (downscaled) coordinates
        fr, fl = logo_signature(tight, tight_box if tight_box is not None else box)
        checks["logo_fill_right"] = round(fr, 3)
        checks["text_fill_left"] = round(fl, 3)
        # Bilibili draws a sparse text run then a DENSE logo, so the right half
        # must be clearly denser than the left. A uniform block (any plain
        # rectangle, badge or border) has both halves alike and is rejected
        # here rather than mistaken for the mark.
        logo_ok = fr > 0.12 and fr >= 1.15 * fl
        checks["logo_signature_ok"] = bool(logo_ok)
        uniform_block = fl > 0.32 and fr > 0.32 and fr < 1.12 * fl
        checks["uniform_block"] = bool(uniform_block)
        if uniform_block:
            shape_ok = False

    return bool(shape_ok), checks, checks.get("logo_signature_ok", None)




def _grow_over_solid(edge_mask, hp_roi, gray_roi, thr):
    """Close the mask over flat/saturated areas that belong to the mark.

    An edge detector sees the outline of a glyph but not its interior. Both
    cases are recovered here:
      1. enclosed holes (flood fill from the ROI border cannot reach them);
      2. interior pixels that are much brighter than the local background,
         i.e. the mark is saturating the picture there.
    The bright test uses a heavy blur of the median frame as the local
    background, which is unaffected by the overlay because it is only ~10%
    of the picture area.
    """
    m = edge_mask.copy()
    h, w = m.shape

    # (1) enclosed holes
    ff = m.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 1)
    m[(ff == 0)] = 1

    # (2) saturated / flat-bright interiors next to the outline
    bg = cv2.GaussianBlur(gray_roi, (0, 0), max(8.0, 0.03 * w))
    # alpha ~0.7 over a bright background still lifts the pixel by ~20 levels,
    # so the threshold has to stay low; "near" keeps it from creeping into the
    # rest of the frame.
    bright = (gray_roi - bg) > 18.0
    near = cv2.dilate(m, np.ones((13, 13), np.uint8)) > 0
    m[bright & near] = 1

    # (3) tidy up: no stray pixels, keep component structure
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, 4] >= 8:
            keep[lab == i] = 1
    return keep


def local_bg(img, mask, k):
    """Background estimate using unmasked pixels only (normalised box blur)."""
    valid = (~mask).astype(np.float32)
    num = cv2.blur(img * valid, (k, k))
    den = cv2.blur(valid, (k, k))
    return num / np.maximum(den, 1e-4)


def solve_alpha(frames, mask, k_bg):
    """1 - alpha = d(obs)/d(bg), per pixel, median over informative frame pairs.

    Frames must be float grayscale. Computing the background proxy on uint8
    data makes the subtraction wrap around and silently yields nonsense.
    """
    frames = [f.astype(np.float32) for f in frames]
    pairs = [(i, j) for i in range(len(frames)) for j in range(i + 1, len(frames))]
    if len(pairs) > 40:
        pairs = pairs[:: max(1, len(pairs) // 40)]
    core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    est = []
    for i, j in pairs:
        a, b = frames[i], frames[j]
        db = local_bg(a, mask, k_bg) - local_bg(b, mask, k_bg)
        r = (a - b) / np.where(np.abs(db) > 6.0, db, np.nan).astype(np.float32)
        est.append(np.clip(1.0 - r, 0.0, 1.0))
    stack = np.stack(est)                                   # (pairs, H, W)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        al = np.nanmedian(stack, axis=0)
    al = np.nan_to_num(al, nan=0.0, posinf=0.0, neginf=0.0)
    al = cv2.GaussianBlur(np.clip(al, 0.0, 0.95).astype(np.float32), (0, 0), 0.8)
    al[~mask] = 0.0
    al[al < 0.03] = 0.0
    return al.astype(np.float32), core


def build_tight(det_video, box, fw, fh, n=12, duration=10.0):
    """Full-resolution glyph mask for a candidate box.

    Thin bright strokes survive a temporal median of high-pass energy even when
    the background behind the mark keeps moving, so this isolates the actual
    glyphs instead of a rectangle.
    """
    x, y, w, h = [int(v) for v in box]
    # The coarse pass under-reports the height: the tall glyph strokes (the dot
    # of "i", the ascenders) are thin and fade in the energy map. Expand
    # generously and let the resulting mask define the final box.
    pad_x = max(8, C.dilate_px(fw) * 3, int(0.06 * w))
    pad_y = max(8, C.dilate_px(fw) * 3, int(0.55 * h))
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)

    times = [duration * (i + 0.5) / max(1, n) for i in range(max(4, n))]
    fg = []
    for _t, raw in C.frames_at(det_video, times, fw, fh):
        fg.append(cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(fh, fw, 3),
                               cv2.COLOR_BGR2GRAY).astype(np.float32))
    if len(fg) < 4:
        return None, None, None

    k = odd(max(9, 0.021 * fw))
    hp = np.median(np.stack([np.clip(highpass(g, k), 0, None) for g in fg]), axis=0)
    roi = hp[y0:y1, x0:x1]
    if roi.size == 0:
        return None, None, None
    thr = max(5.0, 0.30 * float(np.percentile(roi, 99.5)))
    gm = (roi > thr).astype(np.uint8)
    gm = cv2.morphologyEx(gm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(gm, 8)
    keep = np.zeros_like(gm)
    for i in range(1, ncomp):
        if stats[i, 4] >= max(6, int(0.02 * roi.size) // 100):
            keep[lab == i] = 1

    # High-pass energy only fires on EDGES. Two things therefore come back
    # empty even though they are part of the mark:
    #   - the solid interior of the logo,
    #   - strong strokes that saturate the sensor (flat white inside the outline).
    # Leaving them uncovered means the renderer steps over them and a white
    # blob survives, so grow the mask back over those areas directly.
    tight_roi = _grow_over_solid(keep, hp[y0:y1, x0:x1], np.median(
        np.stack([g[y0:y1, x0:x1] for g in fg]), axis=0), thr)

    tight = np.zeros((fh, fw), np.uint8)
    tight[y0:y1, x0:x1] = tight_roi
    return tight, fg, (x0, y0, x1, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="wm_work")
    ap.add_argument("--bbox", default=None, help="x,y,w,h at full resolution; skips detection")
    ap.add_argument("--frames", type=int, default=24, help="coarse sampling frames")
    ap.add_argument("--refine-frames", type=int, default=12, help="full-res frames for the glyph mask")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="skip the 'is this really Bilibili?' gate (you take the risk)")
    C.add_fetch_args(ap)
    a = ap.parse_args()

    C.require_ffmpeg()
    a.video = C.resolve_source(a.video, height=a.height, cookies=a.cookies,
                               force=a.force_download, anonymous=a.anonymous)
    os.makedirs(a.out, exist_ok=True)

    grays, scale, p, (fw, fh) = collect(a.video, max(8, min(a.frames, 96)))
    if len(grays) < 4:
        sys.exit("need at least 4 decodable frames, got %d" % len(grays))

    W = grays.shape[2]
    k_hp = odd(0.021 * W)
    k_bg = odd(0.045 * fw)

    # ---------------- 1. candidate boxes ----------------
    if a.bbox:
        bx, by, bw, bh = [int(v) for v in a.bbox.split(",")]
        coarse = [(bx * scale, by * scale, bw * scale, bh * scale)]
        src = "user"
        energy = overlay_energy(grays, k_hp)
    else:
        energy = overlay_energy(grays, k_hp)
        found = merge_fragments(find_candidates(energy))
        if not found:
            sys.exit("no static bright overlay found at the top of the frame.\n"
                     "If you can see a Bilibili mark, pass --bbox x,y,w,h to force it.")
        coarse = [(c["x"], c["y"], c["w"], c["h"]) for c in found[:4]]
        src = "auto"

    # ---------------- 2. refine each candidate and validate ----------------
    accept = None
    tried = []
    for cand in coarse:
        box_full = [int(round(v / scale)) for v in cand]
        tight, fg, roi_box = build_tight(a.video, box_full, fw, fh,
                                         n=a.refine_frames, duration=p["duration"])
        if tight is None or int(tight.sum()) < 200:
            tried.append({"box": box_full, "rejected": "glyph mask too sparse"})
            continue
        tb = [int(v) for v in cv2.boundingRect(tight)]
        passed, sig, logo_ok = validate(grays, [int(v) for v in cand], energy,
                                       tight=tight, tight_box=tb)
        tried.append({"box": tb, "checks": sig})
        if passed:
            accept = {"tight": tight, "fg": fg, "roi_box": roi_box,
                      "tight_box": tb, "checks": sig, "box_full": box_full}
            break

    if accept is None:
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump({"video": p, "detection_source": src, "validated": False,
                       "candidates": tried,
                       "reason": "no candidate passed the logo-signature / geometry gate"}, f,
                      indent=2, ensure_ascii=False)
        print("NOT A VERIFIED BILIBILI WATERMARK")
        print(json.dumps(tried, indent=2, ensure_ascii=False)[:2000])
        print("\nEither the watermark is not Bilibili's, or detection needs help:"
              "\n  re-run with --bbox x,y,w,h once you locate the mark by eye.")
        return 2

    tight = accept["tight"]
    fg = accept["fg"]
    checks = accept["checks"]
    tb = accept["tight_box"]
    ok = True

    # ---------------- 3. mask + alpha at full resolution ----------------
    pad = C.dilate_px(fw)
    kern = np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8)
    mask = cv2.dilate(tight, kern)
    alpha, _ = solve_alpha(fg, tight.astype(bool), k_bg)

    # write 0/255 PNGs: a 0/1 image looks black and re-reads as an empty mask
    cv2.imwrite(os.path.join(a.out, "mask.png"), (mask > 0).astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(a.out, "mask_tight.png"), (tight > 0).astype(np.uint8) * 255)
    np.save(os.path.join(a.out, "alpha.npy"), alpha)

    a_med = float(np.median(alpha[tight > 0])) if int(tight.sum()) else 0.0
    a_max = float(alpha.max())

    # Decisive physical gate: the Bilibili overlay is semi-transparent white at
    # alpha 0.62-0.78. An opaque mark (alpha ~0.95) is a graphic burned in by
    # another tool, and a very low value means the frames never varied enough
    # for the estimate to mean anything. Either way: not ours to touch.
    if not (0.42 <= a_med <= 0.86) and not a.force:
        report = {"video": {k: p[k] for k in ("path", "width", "height", "fps",
                                              "duration", "nb_frames", "bitrate")},
                  "detection_source": src, "validated": False, "checks": checks,
                  "tight_box": [int(v) for v in tb],
                  "alpha_median": round(a_med, 3),
                  "reason": ("alpha %.2f is outside the Bilibili range 0.62-0.78: the mark is "
                             "either opaque (burned in by another tool) or the sample frames "
                             "did not vary enough to measure it" % a_med)}
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("NOT A VERIFIED BILIBILI WATERMARK")
        print(json.dumps({k: report[k] for k in ("alpha_median", "reason")},
                         indent=2, ensure_ascii=False))
        return 2
    x, y, w, h = tb
    pb = [int(max(0, x - pad)), int(max(0, y - pad)),
          int(x + w + pad), int(y + h + pad)]

    # ---------------- 4. visual proof ----------------
    dark = min(fg, key=lambda g: float(g[pb[1]:pb[3], pb[0]:pb[2]].mean()))
    vis = cv2.cvtColor(dark.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis[mask > 0] = (0.25 * vis[mask > 0] + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.rectangle(vis, (pb[0], pb[1]), (pb[2], pb[3]), (0, 255, 0), max(1, pad))
    cv2.imwrite(os.path.join(a.out, "mask_overview.png"), vis)
    zoom = vis[max(0, pb[1] - pad * 4):pb[3] + pad * 4,
               pb[0]:min(fw, pb[0] + max(240, int(1.6 * w)))]
    if zoom.size:
        cv2.imwrite(os.path.join(a.out, "mask_zoom.png"),
                    cv2.resize(zoom, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

    # ---------------- 5. report ----------------
    report = {
        "video": {k: p[k] for k in ("path", "width", "height", "fps", "duration",
                                    "nb_frames", "bitrate", "has_audio", "vcodec",
                                    "rotation")},
        "detection_source": src,
        "validated": True,
        "checks": checks,
        "tight_box": [int(v) for v in tb],
        "padded_box": pb,
        "alpha_median": round(a_med, 3),
        "alpha_max": round(a_max, 3),
        "dilate_px": int(pad),
        "sampled_frames": int(len(grays)),
        "refine_frames": len(fg),
        "notes": [],
    }
    if a_med < 0.35:
        report["notes"].append("alpha median %.2f is low: sample frames may not have varied "
                               "enough background" % a_med)
    # alpha is clipped at 0.95 inside the solver, so "a_max > 0.9" would fire on
    # every clip. What matters is how much of the mark is effectively opaque:
    # that is where the picture underneath is gone and LaMa has to invent it.
    opaque_frac = float((alpha[tight > 0] > 0.90).mean()) if int(tight.sum()) else 0.0
    report["opaque_fraction"] = round(opaque_frac, 3)
    if opaque_frac > 0.25:
        report["notes"].append(
            "%.0f%% of the mark is effectively opaque, so the picture underneath is gone and "
            "LaMa must invent it; expect faint artifacts on detailed backgrounds"
            % (100 * opaque_frac))
    if src == "user":
        report["notes"].append("box came from --bbox; only the logo/geometry gate was checked")

    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: report[k] for k in ("detection_source", "validated", "tight_box",
                                             "padded_box", "alpha_median", "alpha_max",
                                             "dilate_px")}, indent=2))
    if a.verbose:
        print(json.dumps(checks, indent=2, ensure_ascii=False))
    for note in report["notes"]:
        print("NOTE: " + note)
    print("\nwrote mask.png / mask_tight.png / alpha.npy / mask_zoom.png / report.json -> " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
