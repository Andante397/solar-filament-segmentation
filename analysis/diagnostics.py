"""Out-of-fold diagnostics behind the figures and tables of the technical report.

Every training image is predicted by the fold model that did not see it, and
the decision layer is cross-fitted the same way, so all numbers are honest
estimates. Needs `output/val_prob_fold*.npz` (from `python -m filament.infer`)
and `models/decision_layer_params.json` (from `decision_layer fit`).

    python -m analysis.diagnostics --data-root <competition folder>

Writes to `analysis/out/`:
  diagnostics.json   all numbers
  fig_quality.pdf    IoU / Dice of matched instances, recall vs. radius and area
  fig_examples.pdf   annotations and predictions on a few image windows
  fig_pipeline.pdf   the processing stages on one image
"""
import argparse
import json
import os
import pickle
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_util

from filament import config as C
from filament import decision_layer as dl
from filament.annotations import Annotations
from filament.data import make_planes
from filament.disk import load_disk_cache, read_gray

R_BINS = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]                # centroid distance r/R
AREA_BINS = [0, 250, 500, 1000, 2000, 4000, np.inf]       # annotated area in px


# ---------------------------------------------------------------- matching
def match_reading(pred_lab, n_pred, gt_masks):
    """Compare the emitted instances of an image with one annotator reading.

    A prediction and an annotation match if IoU > 0.5; predictions do not
    overlap, so the match is unique. Unmatched annotations are classified by
    what went wrong:
      missed       less than 10 % of the annotation is covered by predictions
      fragmented   two or more predictions lie mostly inside it
      merged       its best prediction mostly covers another annotation as well
      mislocalised a single partial overlap with IoU <= 0.5
    """
    area_p = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)[1:].astype(np.float64)
    inter = np.zeros((len(gt_masks), n_pred))
    area_g = np.array([g.sum() for g in gt_masks], dtype=np.float64)
    for i, g in enumerate(gt_masks):
        inter[i] = np.bincount(pred_lab[g], minlength=n_pred + 1)[1:]
    iou = inter / np.maximum(area_g[:, None] + area_p[None, :] - inter, 1)
    inside_gt = inter / np.maximum(area_p[None, :], 1) > 0.5       # prediction mostly inside annotation
    covers_gt = inter / np.maximum(area_g[:, None], 1) > 0.5       # annotation mostly inside prediction

    matched_pred = np.zeros(n_pred, bool)
    gts = []
    for i in range(len(gt_masks)):
        j = int(np.argmax(iou[i])) if n_pred else -1
        ok = n_pred > 0 and iou[i, j] > 0.5
        if ok:
            matched_pred[j] = True
            status = "matched"
        elif inter[i].sum() < 0.1 * area_g[i]:
            status = "missed"
        elif inside_gt[i].sum() >= 2:
            status = "fragmented"
        elif covers_gt[:, int(np.argmax(inter[i]))].sum() >= 2:
            status = "merged"
        else:
            status = "mislocalised"
        gts.append(dict(status=status, area=float(area_g[i]),
                        iou=float(iou[i, j]) if ok else 0.0,
                        n_fragments=int(inside_gt[i].sum())))
    preds = []
    for j in range(n_pred):
        if matched_pred[j]:
            status = "matched"
        elif len(gt_masks) == 0 or inter[:, j].sum() < 0.1 * area_p[j]:
            status = "no annotation"
        elif inside_gt[:, j].any():
            status = "fragment"
        elif covers_gt[:, j].sum() >= 2:
            status = "merger"
        else:
            status = "mislocalised"
        preds.append(dict(status=status, n_covered=int(covers_gt[:, j].sum())))
    return gts, preds


def pq_terms(gts, n_pred):
    """(sum of matched IoU, TP, FP, FN) of one reading."""
    tp = sum(g["status"] == "matched" for g in gts)
    return sum(g["iou"] for g in gts), tp, n_pred - tp, len(gts) - tp


def pq(terms):
    """PQ, SQ, RQ from a list of (iou_sum, TP, FP, FN)."""
    s, tp, fp, fn = np.sum(np.asarray(terms, dtype=np.float64).reshape(-1, 4), axis=0)
    den = tp + 0.5 * fp + 0.5 * fn
    return (s / den if den else 0.0, s / tp if tp else 0.0, tp / den if den else 0.0)


def bootstrap(per_image, stat, n_boot=2000, seed=0):
    """Percentile 95 % interval of `stat(list of per-image items)`, resampling images."""
    rng = np.random.default_rng(seed)
    n = len(per_image)
    vals = [stat([per_image[i] for i in rng.integers(0, n, n)]) for _ in range(n_boot)]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def wilson(k, n, z=1.96):
    """Wilson score interval of a proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, centre + half


# ---------------------------------------------------------------- annotator agreement
def human_terms(ann, image_ids):
    """PQ terms of every ordered pair of readings of one image, one reading
    playing the prediction and the other the reference."""
    rles = [[ann.coco.annToRLE(a) for a in ann.instances(i)] for i in image_ids]
    rles = [[mask_util.frPyObjects(r, *r["size"]) if isinstance(r["counts"], list) else r
             for r in rs] for rs in rles]
    terms = []
    for a in range(len(rles)):
        for b in range(len(rles)):
            if a == b:
                continue
            if not rles[a] or not rles[b]:
                terms.append((0.0, 0, len(rles[a]), len(rles[b])))
                continue
            iou = mask_util.iou(rles[a], rles[b], [0] * len(rles[b]))   # (pred, reference)
            pairs = sorted(((iou[i, j], i, j) for i in range(iou.shape[0])
                            for j in range(iou.shape[1]) if iou[i, j] > 0.5), reverse=True)
            used_p, used_g, s = set(), set(), 0.0
            for v, i, j in pairs:              # annotations of one reader may overlap
                if i not in used_p and j not in used_g:
                    used_p.add(i); used_g.add(j); s += float(v)
            tp = len(used_p)
            terms.append((s, tp, len(rles[a]) - tp, len(rles[b]) - tp))
    return terms


# ---------------------------------------------------------------- main analysis
def analyse(paths, model_dir, out_dir, limit=0):
    with open(os.path.join(model_dir, "decision_layer_params.json")) as f:
        params = json.load(f)
    mask_t, tau = params["mask_t"], params["tau"]
    ann = Annotations(paths.annotations)
    disk = load_disk_cache(paths.disk_cache)

    cache = os.path.join(out_dir, f"oof_table_limit{limit}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            table, imgs = pickle.load(f)
    else:
        table, imgs = dl.build_oof_table(paths, mask_t, limit)
        table = dl.cross_fit(table)
        with open(cache, "wb") as f:
            pickle.dump((table, imgs), f)
    table = table.assign(cand_id=table.groupby("obs").cumcount() + 1)
    emitted = {name: g.loc[g["u_hat"] > tau, "cand_id"].to_numpy()
               for name, g in table.groupby("obs")}

    archives = {k: np.load(paths.prob("val", k)) for k in range(C.N_FOLDS)
                if os.path.exists(paths.prob("val", k))}
    per_image, gt_rows, pred_rows = {}, [], []
    for n_done, (name, info) in enumerate(imgs.items(), 1):
        meta = disk[name]
        valid = dl.disk_grid(meta)[2] < C.DISK_FRAC
        lab, _ = dl.label_candidates(archives[info["fold"]][name], valid, mask_t)
        keep = emitted.get(name, np.zeros(0, int))
        remap = np.zeros(lab.max() + 1, np.int32)
        remap[keep] = np.arange(1, len(keep) + 1)
        pred = remap[lab]
        terms = []
        for iid in ann.readings_of[name]:
            masks = ann.instance_masks(iid)
            gts, preds = match_reading(pred, len(keep), masks)
            for g, m in zip(gts, masks):
                ys, xs = np.nonzero(m)
                g["r"] = float(np.hypot(xs.mean() - meta["cx"], ys.mean() - meta["cy"]) / meta["R"])
                g["obs"] = name
            gt_rows += gts
            pred_rows += preds
            terms.append(pq_terms(gts, len(keep)))
        per_image[name] = dict(model=terms, human=human_terms(ann, ann.readings_of[name]),
                               n_readings=len(terms))
        if n_done % 100 == 0:
            print(f"  matched {n_done}/{len(imgs)} images", flush=True)

    # ---- overall quality
    all_items = list(per_image.values())
    model_stat = lambda items: pq([t for it in items for t in it["model"]])[0]
    pq_all, sq_all, rq_all = pq([t for it in all_items for t in it["model"]])
    out = dict(n_images=len(per_image), tau=tau, mask_t=mask_t,
               model=dict(pq=pq_all, sq=sq_all, rq=rq_all, pq_ci95=bootstrap(all_items, model_stat)),
               pq_reported_by_fit=params["oof_pq"])

    # ---- model vs. annotators on images with several readings
    multi = [it for it in all_items if it["n_readings"] > 1]
    if multi:
        human_stat = lambda items: pq([t for it in items for t in it["human"]])[0]
        diff_stat = lambda items: model_stat(items) - human_stat(items)
        out["multi_annotator"] = dict(
            n_images=len(multi), model_pq=model_stat(multi), human_pq=human_stat(multi),
            model_ci95=bootstrap(multi, model_stat), human_ci95=bootstrap(multi, human_stat),
            difference_ci95=bootstrap(multi, diff_stat))

    # ---- matched instances: IoU and Dice
    iou = np.array([g["iou"] for g in gt_rows if g["status"] == "matched"])
    dice = 2 * iou / (1 + iou)
    q = lambda a: {k: float(v) for k, v in zip(("p10", "p25", "median", "p75", "p90"),
                                               np.percentile(a, [10, 25, 50, 75, 90]))} if len(a) else {}
    out["matched"] = dict(n=int(len(iou)), iou_mean=float(iou.mean()) if len(iou) else 0.0,
                          dice_mean=float(dice.mean()) if len(dice) else 0.0, iou=q(iou), dice=q(dice))

    # ---- what happens to annotations and to predictions
    n_gt, n_pred = len(gt_rows), len(pred_rows)
    gt_status = Counter(g["status"] for g in gt_rows)
    pred_status = Counter(p["status"] for p in pred_rows)
    out["annotations"] = dict(n=n_gt, **{k: v / max(n_gt, 1) for k, v in gt_status.items()})
    out["predictions"] = dict(n=n_pred, **{k: v / max(n_pred, 1) for k, v in pred_status.items()})
    out["one_to_many"] = dict(   # one annotation split over several predictions
        annotations=int(sum(g["n_fragments"] >= 2 for g in gt_rows)),
        fraction=float(np.mean([g["n_fragments"] >= 2 for g in gt_rows])) if n_gt else 0.0)
    out["many_to_one"] = dict(   # one prediction covering several annotations
        predictions=int(sum(p["n_covered"] >= 2 for p in pred_rows)),
        fraction=float(np.mean([p["n_covered"] >= 2 for p in pred_rows])) if n_pred else 0.0)

    # ---- recall by position and size
    def recall_by(key, bins):
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = [g for g in gt_rows if lo <= g[key] < hi]
            p, a, b = wilson(sum(g["status"] == "matched" for g in sel), len(sel))
            rows.append(dict(lo=float(lo), hi=None if np.isinf(hi) else float(hi), n=len(sel),
                             recall=p, ci95=[a, b]))
        return rows
    out["recall_by_radius"] = recall_by("r", R_BINS)
    out["recall_by_area"] = recall_by("area", AREA_BINS)

    with open(os.path.join(out_dir, "diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("recall")}, indent=2))
    return out, iou, per_image, emitted, imgs


# ---------------------------------------------------------------- figures
def fig_quality(out, iou, path):
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
    bins = np.linspace(0.5, 1.0, 26)
    ax[0].hist(iou, bins, histtype="stepfilled", alpha=0.6, label="IoU")
    ax[0].hist(2 * iou / (1 + iou), bins, histtype="step", lw=1.2, color="k", label="Dice")
    ax[0].set_xlabel("overlap of matched instances"); ax[0].set_ylabel("count"); ax[0].legend(frameon=False, fontsize=6, loc="upper left")
    for a, key, xlabel in ((ax[1], "recall_by_radius", "distance from disk centre  r/R"),
                           (ax[2], "recall_by_area", "annotated area (px)")):
        rows = [r for r in out[key] if r["n"] > 0]
        x = np.arange(len(rows))
        y = np.array([r["recall"] for r in rows])
        err = np.array([[r["recall"] - r["ci95"][0], r["ci95"][1] - r["recall"]] for r in rows]).T
        a.errorbar(x, y, yerr=err, fmt="o-", ms=3, lw=1, capsize=2)
        a.set_xticks(x)
        a.set_xticklabels([f"{r['lo']:g}–{r['hi']:g}" if r["hi"] is not None else f">{r['lo']:g}"
                           for r in rows], rotation=35, ha="right", fontsize=6)
        a.set_ylim(0, 1); a.set_xlabel(xlabel); a.set_ylabel("recall")
    for a in ax:
        a.tick_params(labelsize=6); a.xaxis.label.set_size(7); a.yaxis.label.set_size(7)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _prediction(paths, name, fold, emitted, mask_t, disk):
    valid = dl.disk_grid(disk[name])[2] < C.DISK_FRAC
    lab, _ = dl.label_candidates(np.load(paths.prob("val", fold))[name], valid, mask_t)
    return np.isin(lab, emitted.get(name, [])) * lab


def _window(mask, size):
    ys, xs = np.nonzero(mask)
    cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (C.FULL // 2, C.FULL // 2)
    y0 = int(np.clip(cy - size // 2, 0, C.FULL - size)); x0 = int(np.clip(cx - size // 2, 0, C.FULL - size))
    return slice(y0, y0 + size), slice(x0, x0 + size)


def fig_examples(paths, ann, disk, imgs, per_image, emitted, mask_t, path, n=4, size=448):
    """Windows of multi-annotator images at the 20th..80th percentile of image PQ:
    annotations of each reader (top) and emitted instances (bottom)."""
    multi = [(pq(it["model"])[0], name) for name, it in per_image.items() if it["n_readings"] > 1]
    multi = sorted(multi) or sorted((pq(it["model"])[0], name) for name, it in per_image.items())
    picks = [multi[int(round(f * (len(multi) - 1)))][1] for f in np.linspace(0.2, 0.8, n)]
    colours = ["#ffcc00", "#00d0ff", "#ff5fa2", "#7CFC00"]
    fig, ax = plt.subplots(2, n, figsize=(7.0, 3.6))
    for c, name in enumerate(picks):
        img = read_gray(os.path.join(paths.train_images, name))
        readings = [ann.instance_masks(i) for i in ann.readings_of[name]]
        union = np.any([m for r in readings for m in r] or [np.zeros((C.FULL, C.FULL), bool)], axis=0)
        win = _window(union, size)
        pred = _prediction(paths, name, imgs[name]["fold"], emitted, mask_t, disk)
        for r in (0, 1):
            ax[r, c].imshow(img[win], cmap="gray", vmin=0, vmax=255); ax[r, c].axis("off")
        for k, masks in enumerate(readings):
            for m in masks:
                if m[win].any():
                    ax[0, c].contour(m[win], [0.5], colors=colours[k % 4], linewidths=0.7)
        for i in np.unique(pred[win])[1:]:
            ax[1, c].contour(pred[win] == i, [0.5], colors="#00ff7f", linewidths=0.7)
        ax[0, c].set_title(f"image PQ {pq(per_image[name]['model'])[0]:.2f}", fontsize=7)
    ax[0, 0].text(-0.06, 0.5, "annotators", transform=ax[0, 0].transAxes, rotation=90,
                  va="center", ha="right", fontsize=7)
    ax[1, 0].text(-0.06, 0.5, "ours", transform=ax[1, 0].transAxes, rotation=90,
                  va="center", ha="right", fontsize=7)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.94, bottom=0.01, wspace=0.03, hspace=0.03)
    fig.savefig(path, dpi=300); plt.close(fig)


def fig_pipeline(paths, disk, imgs, per_image, emitted, mask_t, path, size=640):
    """Processing stages on the image with the median PQ."""
    ranked = sorted((pq(it["model"])[0], name) for name, it in per_image.items())
    name = ranked[len(ranked) // 2][1]
    meta = disk[name]
    img = read_gray(os.path.join(paths.train_images, name))
    prob = np.load(paths.prob("val", imgs[name]["fold"]))[name]
    valid = dl.disk_grid(meta)[2] < C.DISK_FRAC
    cand, _ = dl.label_candidates(prob, valid, mask_t)
    final = np.isin(cand, emitted.get(name, [])) * cand
    contrast = make_planes(img, meta)[0][0]
    win = _window(cand > 0, size)
    fig, ax = plt.subplots(1, 5, figsize=(7.0, 1.6))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("(a) H-alpha image, fitted limb", fontsize=6)
    t = np.linspace(0, 2 * np.pi, 720)
    ax[0].plot(meta["cx"] + meta["R"] * np.cos(t), meta["cy"] + meta["R"] * np.sin(t), "c-", lw=0.4)
    ax[0].add_patch(plt.Rectangle((win[1].start, win[0].start), size, size, fill=False, ec="y", lw=0.6))
    ax[1].imshow(contrast[win], cmap="gray", vmin=-1.5, vmax=1); ax[1].set_title("(b) limb-corrected contrast", fontsize=6)
    ax[2].imshow(prob[win], cmap="magma", vmin=0, vmax=255); ax[2].set_title("(c) probability", fontsize=6)
    for a, lab, title in ((ax[3], cand, "(d) candidates"), (ax[4], final, "(e) emitted instances")):
        a.imshow(img[win], cmap="gray", vmin=0, vmax=255)
        a.imshow(np.ma.masked_equal(lab[win], 0) % 20, cmap="tab20", alpha=0.75, interpolation="nearest")
        a.set_title(title, fontsize=6)
    for a in ax:
        a.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01, wspace=0.03)
    fig.savefig(path, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work-dir", default="output")
    ap.add_argument("--disk-cache", default="data/disk_cache_v2.json")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default="analysis/out")
    ap.add_argument("--limit", type=int, default=0, help="first N images per fold (smoke test)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    paths = C.Paths(args.data_root, work_dir=args.work_dir, disk_cache=args.disk_cache)
    out, iou, per_image, emitted, imgs = analyse(paths, args.model_dir, args.out_dir, args.limit)

    ann = Annotations(paths.annotations)
    disk = load_disk_cache(paths.disk_cache)
    fig_quality(out, iou, os.path.join(args.out_dir, "fig_quality.pdf"))
    fig_examples(paths, ann, disk, imgs, per_image, emitted, out["mask_t"],
                 os.path.join(args.out_dir, "fig_examples.pdf"))
    fig_pipeline(paths, disk, imgs, per_image, emitted, out["mask_t"],
                 os.path.join(args.out_dir, "fig_pipeline.pdf"))
    print(f"figures and diagnostics.json written to {args.out_dir}/")


if __name__ == "__main__":
    main()
