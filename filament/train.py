"""Train the segmentation network of one cross-validation fold.

Not needed to reproduce the submission (trained weights are in `weights/`).
Training on a GPU is not bit-reproducible, so retrained weights give slightly
different probability maps and scores.

    python -m filament.train --data-root <competition folder> --fold 0
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .annotations import Annotations, assign_folds
from .data import FilamentTiles, ImageStore
from .disk import build_disk_cache, check_radii
from .model import FilamentUNet


def loss_fn(logit, target, valid):
    """BCE + 0.3 Dice, evaluated on on-disk pixels only.

    BCE is a proper scoring rule: with one training sample per annotator
    reading, the network converges to the probability that a random annotator
    marks a pixel, which is the quantity the post-processing thresholds. The
    low-weight Dice term only counteracts the ~0.3 % positive-pixel rate."""
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logit) * valid
    t = target * valid
    dice = 1 - (2 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)
    return bce + C.DICE_WEIGHT * dice, bce.detach(), dice.detach()


def train_fold(paths, fold, epochs=C.EPOCHS, steps_per_epoch=C.STEPS_PER_EPOCH,
               num_workers=4, pretrained_encoder=True):
    """Train on all readings outside `fold` and save the final-epoch weights.

    The last epoch is always kept; the held-out fold is never used to select a
    checkpoint, so out-of-fold predictions stay unbiased."""
    seed = C.SEED + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True         # fixed input shape: autotuning pays off
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    paths.makedirs()
    os.makedirs(paths.weights_dir, exist_ok=True)

    ann = Annotations(paths.annotations)
    folds = assign_folds(ann.observations)
    readings = [r for r in ann.readings if folds[r["file"]] != fold]
    disk = build_disk_cache([paths.train_images], paths.disk_cache)
    check_radii(disk, [r["file"] for r in readings])
    images = ImageStore(paths.train_images, os.path.join(paths.cache_dir, "train_u8.dat"))
    print(f"[fold {fold}] {len(readings)} training readings of "
          f"{len({r['file'] for r in readings})} images | device {device}", flush=True)

    dataset = FilamentTiles(readings, ann, images, disk)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=steps_per_epoch)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler,
                                         num_workers=num_workers, pin_memory=True,
                                         persistent_workers=num_workers > 0)

    model = FilamentUNet(pretrained_encoder).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C.LR, pct_start=0.1,
                                                total_steps=epochs * steps_per_epoch)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for ep in range(epochs):
        model.train()
        t0, agg = time.time(), np.zeros(3)
        for xb, yb, vb in loader:
            # (1, K, C, H, W) -> (K, C, H, W): the K tiles of one image form the batch
            xb = xb.flatten(0, 1).to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.flatten(0, 1).to(device, non_blocking=True)
            vb = vb.flatten(0, 1).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, bce, dice = loss_fn(model(xb), yb, vb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            agg += np.array([loss.item(), bce.item(), dice.item()])
        agg /= max(1, len(loader))
        print(f"[fold {fold}] epoch {ep + 1:2d}/{epochs} | loss {agg[0]:.4f} "
              f"(bce {agg[1]:.4f}, dice {agg[2]:.4f}) | lr {sched.get_last_lr()[0]:.2e} "
              f"| {time.time() - t0:.0f}s", flush=True)

    torch.save(model.state_dict(), paths.weights(fold))
    print(f"[fold {fold}] saved {paths.weights(fold)}", flush=True)
    return paths.weights(fold)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--weights-dir", default="weights_retrained",
                    help="output folder; the default does not overwrite the shipped weights")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--disk-cache", default="data/disk_cache_v2.json")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=C.STEPS_PER_EPOCH)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="do not initialise the encoder from ImageNet weights")
    args = ap.parse_args()
    paths = C.Paths(args.data_root, cache_dir=args.cache_dir, weights_dir=args.weights_dir,
                    disk_cache=args.disk_cache)
    train_fold(paths, args.fold, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
               num_workers=args.num_workers, pretrained_encoder=not args.no_pretrained)


if __name__ == "__main__":
    main()
