"""Solar disk localisation and centre-to-limb intensity profile.

Every later step depends on the disk geometry: the limb-darkening correction,
the r/R input channel, and the on-disk mask. A thresholding approach (Otsu +
area) over-estimates the radius by up to ~100 px on these images because a
bright annulus of variable intensity surrounds the disk out to the field stop
at r ~ 1035 px. The limb itself shows up as a thin bright ring, i.e. a sharp
*rise* in intensity when moving outwards, so we locate that edge directly:

  1. Coarse fit: 4x downsampling -> Canny edges -> Hough circle transform with
     the radius restricted to [R_MIN, R_MAX], which excludes the field stop.
     The Hough transform does not depend on a centroid, so it is robust to an
     off-centre or partly clipped disk.
  2. Refine the radius on the azimuthally averaged radial profile.
  3. Along 720 rays, find the steepest intensity rise near the current radius.
  4. Robust algebraic circle fit with iterative outlier rejection (clouds,
     prominences and clipped rays produce bad edge points).
  5. Repeat 3-4 with a narrower search window.

On the 887 competition images this gives R = 898.3 +/- 0.7 px.
"""
import json
import math
import os
import time

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, uniform_filter1d

from . import config as C

Image.MAX_IMAGE_PIXELS = None

R_MIN, R_MAX = 820, 1008   # radius search range in px; must stay below the field stop


def _hough_coarse(img, scale=4):
    """Hough circle on a downsampled image. Downsampling averages out the
    few-pixel-wide filament edges while keeping the large-scale limb step."""
    from skimage.feature import canny
    from skimage.transform import downscale_local_mean, hough_circle, hough_circle_peaks

    small = downscale_local_mean(img, (scale, scale))
    edges = canny(small, sigma=1.5, low_threshold=0.90, high_threshold=0.97,
                  use_quantiles=True)
    radii = np.arange(R_MIN // scale, R_MAX // scale + 1)
    acc = hough_circle(edges, radii)
    _, cxs, cys, rs = hough_circle_peaks(acc, radii, total_num_peaks=1)
    off = (scale - 1) / 2.0    # pixel-centre offset between the two grids
    return cxs[0] * scale + off, cys[0] * scale + off, float(rs[0] * scale)


def _azimuthal_profile(img, cx, cy, rmax=1080, step=2):
    """Mean intensity as a function of integer radius."""
    sub = img[::step, ::step]
    yy, xx = np.mgrid[0:img.shape[0]:step, 0:img.shape[1]:step]
    r = np.hypot(xx - cx, yy - cy).astype(np.int32)
    m = r < rmax
    s = np.bincount(r[m], weights=sub[m], minlength=rmax)
    c = np.bincount(r[m], minlength=rmax)
    return uniform_filter1d(s / np.maximum(c, 1), 3)


def _limb_radius(profile, r_lo, r_hi):
    """Radius of the limb within [r_lo, r_hi) on a radial profile.

    The limb is the rise into the bright ring. If the steepest change in the
    window is instead a drop (the ring falling back to a dark surround), take
    the strongest rise within 10 px inside of it, provided it is comparable in
    strength; otherwise keep the position of the steepest change."""
    g = np.gradient(profile)
    r_lo, r_hi = max(1, int(r_lo)), min(len(profile) - 1, int(r_hi))
    gw = g[r_lo:r_hi]
    k = int(np.argmax(np.abs(gw)))
    if gw[k] > 0:
        return float(r_lo + k)
    lo = max(0, k - 10)
    kr = lo + int(np.argmax(gw[lo:k])) if k > lo else k
    if kr < k and gw[kr] > 0.5 * abs(gw[k]):
        return float(r_lo + kr)
    return float(r_lo + k)


def _edge_points(img, cx, cy, r0, half, nray=720, dr=0.5):
    """Steepest intensity rise along each ray within r0 +/- half."""
    th = np.linspace(0, 2 * np.pi, nray, endpoint=False)
    rr = np.arange(r0 - half, r0 + half, dr)
    X = cx + np.outer(np.cos(th), rr)
    Y = cy + np.outer(np.sin(th), rr)
    # Rays leaving the frame are discarded so the image border is never
    # mistaken for the limb.
    inside = (X >= 0) & (X <= img.shape[1] - 1) & (Y >= 0) & (Y <= img.shape[0] - 1)
    prof = map_coordinates(img, [Y.ravel(), X.ravel()], order=1,
                           mode="nearest").reshape(X.shape)
    prof = uniform_filter1d(prof, 5, axis=1)
    g = np.gradient(prof, dr, axis=1)
    k = np.argmax(g, 1)
    strength = np.abs(g[np.arange(nray), k])
    ok = inside.all(1) & (k > 2) & (k < len(rr) - 3)
    if ok.any():
        ok &= strength > 0.3 * np.median(strength[ok])
    re = rr[k]
    return cx + re * np.cos(th), cy + re * np.sin(th), ok


def _fit_circle(x, y):
    """Algebraic least-squares circle."""
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    (a, c, d), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(a), float(c), float(math.sqrt(d + a * a + c * c))


def _robust_fit(x, y, n_iter=8, k=3.0):
    """Circle fit with iterative rejection of points beyond k robust sigmas."""
    keep = np.ones(len(x), bool)
    for _ in range(n_iter):
        cx, cy, R = _fit_circle(x[keep], y[keep])
        res = np.hypot(x - cx, y - cy) - R
        s = 1.4826 * np.median(np.abs(res[keep]))
        new = np.abs(res) < max(k * s, 1.0)
        if new.sum() < 50 or (new == keep).all():
            break
        keep = new
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    return cx, cy, R, keep, rms


def detect_disk(img_u8, nray=720):
    """Fit the solar disk. Returns dict(cx, cy, R, rms, inlier_frac), where
    `rms` is the residual of the final fit in px and `inlier_frac` the fraction
    of rays used; both are quality flags."""
    img = np.asarray(img_u8, dtype=np.float32)
    cx, cy, r_hough = _hough_coarse(img)
    profile = _azimuthal_profile(img, cx, cy)
    R = _limb_radius(profile, r_hough - 25, min(r_hough + 25, R_MAX + 10))

    n_keep, rms = 0, float("nan")
    for half in (16, 8):
        x, y, ok = _edge_points(img, cx, cy, R, half, nray=nray)
        if ok.sum() < 50:
            break
        cx, cy, R, keep, rms = _robust_fit(x[ok], y[ok])
        n_keep = int(keep.sum())
    return dict(cx=cx, cy=cy, R=R, rms=rms, inlier_frac=n_keep / nray)


def limb_profile(img_u8, cx, cy, R, nbins=C.NBINS, step=C.LIMB_STEP):
    """Median intensity in `nbins` radial shells covering r/R in [0, 1).

    The median is used instead of the mean because filaments are dark outliers:
    with a mean they would lower the reference level of their own shell and
    partly normalise themselves away."""
    H, W = img_u8.shape
    sub = img_u8[::step, ::step]
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
    inside = r < 0.999
    rb = np.clip((r[inside] * nbins).astype(np.int32), 0, nbins - 1)
    vals = sub[inside].astype(np.float32)

    order = np.argsort(rb, kind="stable")
    rb_s, vals_s = rb[order], vals[order]
    bounds = np.searchsorted(rb_s, np.arange(nbins + 1))
    prof = np.full(nbins, np.nan, dtype=np.float32)
    for b in range(nbins):
        lo, hi = bounds[b], bounds[b + 1]
        if hi > lo:
            prof[b] = np.median(vals_s[lo:hi])

    # Fill empty shells and smooth lightly (the outermost shells are noisy).
    idx = np.arange(nbins)
    ok = ~np.isnan(prof)
    prof = np.interp(idx, idx[ok], prof[ok]).astype(np.float32)
    return uniform_filter1d(prof, size=9, mode="nearest")


def radius_map(meta, x0=0, y0=0, size=C.FULL):
    """r/R for a square window with top-left corner (x0, y0), float32."""
    dx = np.arange(x0, x0 + size, dtype=np.float32) - np.float32(meta["cx"])
    dy = np.arange(y0, y0 + size, dtype=np.float32) - np.float32(meta["cy"])
    return np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2) / np.float32(meta["R"])


def disk_mask(meta):
    """Boolean full-frame mask of on-disk pixels."""
    yy, xx = np.ogrid[0:C.FULL, 0:C.FULL]
    return (xx - meta["cx"]) ** 2 + (yy - meta["cy"]) ** 2 < (C.DISK_FRAC * meta["R"]) ** 2


def read_gray(path):
    return np.array(Image.open(path).convert("L"))


def load_disk_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def build_disk_cache(image_dirs, path, verbose=True):
    """Disk parameters and limb profile for every image in `image_dirs`,
    stored as JSON keyed by file name. Existing entries are reused."""
    cache = load_disk_cache(path)
    changed, t0 = False, time.time()
    for folder in image_dirs:
        todo = [f for f in C.list_images(folder) if f not in cache]
        for i, name in enumerate(todo):
            img = read_gray(os.path.join(folder, name))
            d = detect_disk(img)
            cache[name] = {"cx": d["cx"], "cy": d["cy"], "R": d["R"], "rms": d["rms"],
                           "prof": limb_profile(img, d["cx"], d["cy"], d["R"]).tolist()}
            changed = True
            if verbose and (i + 1) % 100 == 0:
                print(f"  disk fit {i + 1}/{len(todo)} in {folder} ({time.time() - t0:.0f}s)",
                      flush=True)
    if changed:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"      # atomic replace: readers never see a partial file
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    return cache


def check_radii(cache, names=None):
    """Raise if any fitted radius is implausible; return (mean, std)."""
    radii = np.array([cache[n]["R"] for n in (names if names is not None else cache)])
    bad = np.abs(radii - C.EXPECTED_RADIUS) >= C.RADIUS_TOL
    if bad.any():
        raise ValueError(f"implausible disk radius: {radii.min():.1f} - {radii.max():.1f} px")
    return float(radii.mean()), float(radii.std())
