"""Full-disk inference: overlapping tiles, cosine blending, dihedral TTA."""
import argparse
import os
import time

import numpy as np
import torch

from . import config as C
from .annotations import Annotations, assign_folds
from .data import ImageStore, make_planes
from .disk import build_disk_cache, check_radii, disk_mask
from .model import load_model


def _blend_window(size):
    """Raised-cosine weights: tiles are averaged smoothly, so no seams appear
    at tile borders (a seam can cut a filament in two)."""
    w = np.hanning(size + 2)[1:-1].astype(np.float32)
    return np.maximum(np.outer(w, w), 1e-3)


def _tta_variants(n):
    """(hflip, vflip, transpose) combinations."""
    flips = [(0, 0), (1, 0), (0, 1), (1, 1)]
    if n <= 1:
        return [(0, 0, 0)]
    if n <= 4:
        return [(h, v, 0) for h, v in flips]
    return [(h, v, t) for t in (0, 1) for h, v in flips]


@torch.no_grad()
def predict_full(img_u8, meta, model, device, tta=C.TTA, batch=8):
    """Probability map (float32, FULL x FULL), zero outside the disk.

    The image is not resampled: a 512-px window slides with stride 256, every
    window is predicted under `tta` symmetries, and all predictions are
    averaged with raised-cosine weights."""
    S = C.TILE
    origins = list(range(0, C.FULL - S + 1, C.STRIDE))
    if origins[-1] != C.FULL - S:
        origins.append(C.FULL - S)

    acc = np.zeros((C.FULL, C.FULL), np.float32)
    wacc = np.zeros((C.FULL, C.FULL), np.float32)
    win = _blend_window(S)
    variants = _tta_variants(tta)
    use_amp = device.type == "cuda"
    coords, buf = [], []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(device).to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = torch.sigmoid(model(x)).float().cpu().numpy()[:, 0]
        for (y0, x0, h, v, t), pr in zip(coords, out):
            if t:                                  # undo the transform
                pr = pr.T
            if v:
                pr = pr[::-1, :]
            if h:
                pr = pr[:, ::-1]
            acc[y0:y0 + S, x0:x0 + S] += pr * win
            wacc[y0:y0 + S, x0:x0 + S] += win
        coords.clear()
        buf.clear()

    for y0 in origins:
        for x0 in origins:
            planes, _ = make_planes(img_u8, meta, x0, y0, S)
            for h, v, t in variants:
                p = planes
                if h:
                    p = p[:, :, ::-1]
                if v:
                    p = p[:, ::-1, :]
                if t:
                    p = p.transpose(0, 2, 1)
                buf.append(np.ascontiguousarray(p))
                coords.append((y0, x0, h, v, t))
                if len(buf) >= batch:
                    flush()
    flush()

    prob = acc / np.maximum(wacc, 1e-6)
    return (prob * disk_mask(meta)).astype(np.float32)


def to_uint8(prob):
    """Probabilities are stored as uint8 (1/255 resolution)."""
    return np.clip(prob * 255 + 0.5, 0, 255).astype(np.uint8)


def infer_images(names, images, disk, model, device, tta=C.TTA, label=""):
    """Probability maps (uint8) for a list of images, keyed by file name."""
    probs, t0 = {}, time.time()
    for i, name in enumerate(names):
        probs[name] = to_uint8(predict_full(images[name], disk[name], model, device, tta=tta))
        if (i + 1) % 20 == 0 or i + 1 == len(names):
            el = time.time() - t0
            print(f"  {label} {i + 1}/{len(names)}  {el:.0f}s "
                  f"(~{el / (i + 1) * (len(names) - i - 1) / 60:.0f} min left)", flush=True)
    return probs


def set_deterministic():
    """Prefer reproducible convolution algorithms over the fastest ones."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_fold(paths, fold, splits=("val", "test"), tta=C.TTA, limit=0, device=None):
    """Predict the out-of-fold training images ('val') and/or the test images
    with the model of one fold and save them as compressed npz files."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths.makedirs()
    model = load_model(paths.weights(fold), device)
    for split in splits:
        src = "train" if split == "val" else "test"
        images = ImageStore(paths.image_dir(src), os.path.join(paths.cache_dir, f"{src}_u8.dat"))
        if split == "val":
            folds = assign_folds(Annotations(paths.annotations).observations)
            names = [n for n in sorted(folds) if folds[n] == fold]
        else:
            names = images.names
        if limit:
            names = names[:limit]
        disk = build_disk_cache([paths.image_dir(src)], paths.disk_cache)
        check_radii(disk, names)
        probs = infer_images(names, images, disk, model, device, tta=tta,
                             label=f"[fold {fold}] {split}")
        np.savez_compressed(paths.prob(split, fold), **probs)
        print(f"[fold {fold}] saved {paths.prob(split, fold)} ({len(probs)} images)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(C.N_FOLDS)))
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--work-dir", default="output")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--disk-cache", default="data/disk_cache_v2.json")
    ap.add_argument("--tta", type=int, default=C.TTA)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    args = ap.parse_args()
    paths = C.Paths(args.data_root, args.work_dir, args.cache_dir, args.weights_dir, args.disk_cache)
    set_deterministic()
    for fold in args.folds:
        run_fold(paths, fold, splits=args.splits, tta=args.tta, limit=args.limit)


if __name__ == "__main__":
    main()
