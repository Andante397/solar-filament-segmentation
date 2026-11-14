"""Image access, network input planes and the training dataset."""
import math
import os
import random
import time

import numpy as np
import torch
from pycocotools import mask as mask_util

from . import config as C
from .disk import radius_map, read_gray


# ---------------------------------------------------------------- images
class ImageStore:
    """Decoded 8-bit images of one split as a single memory-mapped file.

    Decoding a 2048x2048 JPEG for every 512x512 training tile would dominate
    the training loop. With a memmap, reading a tile touches only its pages.
    """

    def __init__(self, image_dir, cache_path, verbose=True):
        self.names = C.list_images(image_dir)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.path = cache_path
        shape = (len(self.names), C.FULL, C.FULL)
        if not os.path.exists(cache_path) or os.path.getsize(cache_path) != int(np.prod(shape)):
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            t0 = time.time()
            mm = np.memmap(cache_path, np.uint8, "w+", shape=shape)
            for i, name in enumerate(self.names):
                mm[i] = read_gray(os.path.join(image_dir, name))
            mm.flush()
            del mm
            if verbose:
                print(f"decoded {len(self.names)} images -> {cache_path} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        self._shape = shape
        self._mm = None

    def __getitem__(self, name):
        """Read-only view of one image (not a copy)."""
        if self._mm is None:      # opened lazily so each DataLoader worker maps the file itself
            self._mm = np.memmap(self.path, np.uint8, "r", shape=self._shape)
        return self._mm[self.index[name]]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        return state


# ---------------------------------------------------------------- input planes
def limb_flat(r, prof):
    """Limb-darkening reference intensity at normalised radius `r`."""
    ridx = np.clip(r * C.NBINS, 0, C.NBINS - 1)
    return np.interp(ridx, np.arange(C.NBINS), prof).astype(np.float32)


def make_planes(img_u8, meta, x0=0, y0=0, size=C.FULL):
    """Three input planes and the on-disk mask for one window.

      0. limb-corrected contrast I / F(r) - 1: how much darker than the local
         quiet-Sun level a pixel is, independent of its distance from disk centre
      1. raw intensity: keeps absolute brightness (sunspots, plage)
      2. normalised radius r/R: tells the network how strong the foreshortening is

    Geometric augmentation must transform plane 2 together with the others;
    photometric augmentation must leave it untouched.
    """
    prof = np.asarray(meta["prof"], dtype=np.float32)
    tile = np.asarray(img_u8[y0:y0 + size, x0:x0 + size], dtype=np.float32)
    r = radius_map(meta, x0, y0, size)

    contrast = tile / np.maximum(limb_flat(r, prof), 1e-3) - 1.0
    p0 = np.clip(contrast, -0.6, 0.3) / 0.3      # filaments are on the negative side
    p1 = tile / 127.5 - 1.0
    p2 = r - 0.5

    valid = (r < C.DISK_FRAC).astype(np.float32)
    p0 *= valid
    p1 *= valid                                   # off-disk intensity is zeroed; r is kept
    return np.stack([p0, p1, p2], 0).astype(np.float32), valid


# ---------------------------------------------------------------- training dataset
class FilamentTiles(torch.utils.data.Dataset):
    """One item = one annotator reading -> `k` random tiles of that image.

    Sampling per reading (rather than per image with merged labels) makes the
    BCE-trained network estimate P(a randomly chosen annotator marks the pixel).

    Augmentation is restricted to the 8 symmetries of the square. They are
    exact pixel permutations, so they do not blur structures that are only a
    few pixels wide, and a flipped or rotated solar disk is still a valid one.
    """

    def __init__(self, readings, annotations, images, disk, tile=C.TILE, k=C.TILES_PER_ITEM):
        self.readings = readings
        self.images = images
        self.disk = disk
        self.tile = tile
        self.k = k
        self.union = {r["image_id"]: annotations.union_rle(r["image_id"]) for r in readings}
        self.centres = {r["image_id"]: annotations.centres(r["image_id"]) for r in readings}

    def __len__(self):
        return len(self.readings)

    def _origin(self, centres, meta, rs):
        S = self.tile
        if centres and rs.random() < C.POS_TILE_FRAC:
            fx, fy = centres[rs.randrange(len(centres))]
            jitter = S // 3
            x0 = int(fx) - S // 2 + rs.randint(-jitter, jitter)
            y0 = int(fy) - S // 2 + rs.randint(-jitter, jitter)
        else:                                     # uniform over the disk
            ang = rs.uniform(0, 2 * math.pi)
            rad = meta["R"] * math.sqrt(rs.random()) * 0.98
            x0 = int(meta["cx"] + rad * math.cos(ang)) - S // 2
            y0 = int(meta["cy"] + rad * math.sin(ang)) - S // 2
        return max(0, min(C.FULL - S, x0)), max(0, min(C.FULL - S, y0))

    def __getitem__(self, idx):
        rec = self.readings[idx]
        meta = self.disk[rec["file"]]
        img = self.images[rec["file"]]
        rle = self.union[rec["image_id"]]
        full_mask = (mask_util.decode(rle) if rle is not None
                     else np.zeros((C.FULL, C.FULL), np.uint8))
        centres = self.centres[rec["image_id"]]
        rs = random.Random(random.randrange(1 << 30))

        xs, ys, vs = [], [], []
        for _ in range(self.k):
            x0, y0 = self._origin(centres, meta, rs)
            planes, valid = make_planes(img, meta, x0, y0, self.tile)
            m = full_mask[y0:y0 + self.tile, x0:x0 + self.tile].astype(np.float32)

            op = rs.randrange(8)                  # dihedral transform, applied to all arrays
            if op & 1:
                planes, m, valid = planes[:, :, ::-1], m[:, ::-1], valid[:, ::-1]
            if op & 2:
                planes, m, valid = planes[:, ::-1, :], m[::-1, :], valid[::-1, :]
            if op & 4:
                planes, m, valid = planes.transpose(0, 2, 1), m.T, valid.T
            planes, m, valid = (np.ascontiguousarray(a) for a in (planes, m, valid))
            # photometric jitter on the two intensity planes only
            planes[0] = planes[0] * rs.uniform(0.9, 1.1) + rs.uniform(-0.05, 0.05)
            planes[1] = planes[1] * rs.uniform(0.9, 1.1) + rs.uniform(-0.05, 0.05)

            xs.append(planes)
            ys.append(m[None])
            vs.append(valid[None])
        return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
                torch.from_numpy(np.stack(vs)))
