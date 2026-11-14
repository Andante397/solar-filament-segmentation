"""Decision layer: choose which candidate regions to output so as to maximise PQ.

A candidate is a connected region of the thresholded probability map. The
competition metric is

    PQ = sum(IoU of matched pairs) / (TP + FP/2 + FN/2),

scored against every annotator reading of an image. Emitting one more
candidate adds 1/2 to the denominator per reading (it is either a new TP that
removes an FN, or a new FP) and adds its matched IoU to the numerator. With
h = P(an annotator drew this object and it matches) and m = E[IoU | matched],
emitting a candidate raises PQ iff

    u = h * m > PQ / 2.

h and m are estimated from ~20 per-candidate features with two gradient-boosted
models: a classifier trained on (candidate, reading) pairs and a regressor
trained on the matched pairs. The threshold tau is tuned on out-of-fold
predictions and lands at PQ/2 as predicted.

The idea of scoring candidates by expected PQ gain follows the public
reference solution of this competition (see README); the features,
cross-fitted evaluation and count-matching calibration are ours.

Because candidates do not overlap, a match at IoU > 0.5 is unique, so whether
a candidate matches a reading does not depend on which other candidates are
emitted. PQ of any selection can therefore be computed from per-candidate
sums in milliseconds, which makes the threshold search and bootstrap cheap.

    python -m filament.decision_layer fit     --data-root <competition folder>
    python -m filament.decision_layer predict --data-root <competition folder>
"""
import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import sklearn
from pycocotools import mask as mask_util
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import config as C
from .annotations import Annotations
from .disk import load_disk_cache, read_gray

MIN_CAND_AREA = 50     # px; smaller regions are never candidates
RING = 15              # px; width of the annulus used for the local background

FEATURES = ["log_area", "maxp", "meanp", "log_sump", "frac_p05", "frac_p09", "log_core99",
            "elong", "log_length", "width", "bbox_fill", "rho", "abs_y",
            "con_mean", "con_min", "con_ring", "con_diff", "inten_rel",
            "n_cand_img", "log_nn_dist", "area_rank"]


# ---------------------------------------------------------------- candidates and features
def disk_grid(meta):
    """Pixel coordinates and normalised radius r/R (float32)."""
    yy, xx = np.mgrid[0:C.FULL, 0:C.FULL].astype(np.float32)
    rr = np.hypot(xx - meta["cx"], yy - meta["cy"]) / meta["R"]
    return xx, yy, rr


def contrast_map(img, meta, rr):
    """Limb-corrected contrast I / F(r) - 1, zero outside the disk."""
    prof = np.asarray(meta["prof"], np.float32)
    nb = len(prof)
    flat = np.interp(np.clip(rr * nb, 0, nb - 1), np.arange(nb), prof)
    c = img.astype(np.float32) / np.maximum(flat, 1e-3) - 1.0
    return np.where(rr < C.DISK_FRAC, c, 0.0).astype(np.float32)


def label_candidates(prob_u8, valid, mask_t):
    """Connected on-disk regions with probability > mask_t and area >=
    MIN_CAND_AREA, labelled 1..n in raster order. Returns (labels, n)."""
    lab, n = ndi.label((prob_u8 > int(np.floor(mask_t * 255))) & valid)
    if n == 0:
        return lab, 0
    area = np.bincount(lab.ravel(), minlength=n + 1)
    keep = area >= MIN_CAND_AREA
    keep[0] = False
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    return remap[lab], int(keep.sum())


def candidate_features(prob_u8, img, meta, mask_t):
    """Label map and one feature row per candidate of an image."""
    xx, yy, rr = disk_grid(meta)
    valid = rr < C.DISK_FRAC
    lab, n = label_candidates(prob_u8, valid, mask_t)
    if n == 0:
        return lab, pd.DataFrame(columns=FEATURES + ["area"])
    L = lab.ravel()
    on = L > 0
    Li = L[on]
    idx = np.arange(1, n + 1)
    p = prob_u8.ravel()[on].astype(np.float32) / 255.0

    def per_cand(w=None):
        return np.bincount(Li, weights=w, minlength=n + 1)[1:]

    # probability statistics
    A = per_cand()
    sump = per_cand(p)
    maxp = ndi.maximum(prob_u8, lab, idx).astype(np.float32) / 255.0
    f05 = per_cand((p > 0.5).astype(np.float32)) / A
    f09 = per_cand((p > 0.9).astype(np.float32)) / A
    core = per_cand((p > 0.99).astype(np.float32))

    # shape from second moments
    x, y = xx.ravel()[on], yy.ravel()[on]
    mx, my = per_cand(x) / A, per_cand(y) / A
    sxx = per_cand(x * x) / A - mx ** 2
    syy = per_cand(y * y) / A - my ** 2
    sxy = per_cand(x * y) / A - mx * my
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-3)
    length = 4 * np.sqrt(np.maximum(l1, 1e-3))   # ~ length of a thin uniform strip
    boxes = ndi.find_objects(lab)
    bbox_area = np.array([(s[0].stop - s[0].start) * (s[1].stop - s[1].start) for s in boxes],
                         np.float32)

    # position on the disk
    rho = np.hypot(mx - meta["cx"], my - meta["cy"]) / meta["R"]
    abs_y = np.abs(my - meta["cy"]) / meta["R"]   # solar north is up: proxy for |latitude|

    # image contrast inside the candidate and in a surrounding annulus
    con = contrast_map(img, meta, rr)
    con_mean = per_cand(con.ravel()[on]) / A
    con_min = ndi.minimum(con, lab, idx).astype(np.float32)
    dil = ndi.grey_dilation(lab, size=(2 * RING + 1, 2 * RING + 1))
    ring = (dil > 0) & (lab == 0) & valid
    ring_lab = dil[ring]
    ring_n = np.bincount(ring_lab, minlength=n + 1)[1:]
    ring_s = np.bincount(ring_lab, weights=con[ring], minlength=n + 1)[1:]
    con_ring = np.where(ring_n > 0, ring_s / np.maximum(ring_n, 1), 0.0)
    disk_median = float(np.median(img[valid])) if valid.any() else 1.0
    inten_rel = (per_cand(img.ravel()[on].astype(np.float32)) / A) / max(disk_median, 1.0)

    # context: other candidates in the same image
    if n > 1:
        dd = np.hypot(mx[:, None] - mx[None, :], my[:, None] - my[None, :])
        np.fill_diagonal(dd, np.inf)
        nn_dist = dd.min(1)
    else:
        nn_dist = np.full(n, 3000.0)
    area_rank = (np.argsort(np.argsort(-A)) + 1) / n

    df = pd.DataFrame(dict(
        log_area=np.log(A), maxp=maxp, meanp=sump / A, log_sump=np.log1p(sump),
        frac_p05=f05, frac_p09=f09, log_core99=np.log1p(core),
        elong=np.sqrt(l1 / l2), log_length=np.log(length), width=A / length,
        bbox_fill=A / np.maximum(bbox_area, 1), rho=rho, abs_y=abs_y,
        con_mean=con_mean, con_min=con_min, con_ring=con_ring, con_diff=con_mean - con_ring,
        inten_rel=inten_rel, n_cand_img=np.full(n, n, np.float32),
        log_nn_dist=np.log1p(nn_dist), area_rank=area_rank))
    df["area"] = A
    return lab, df


def match_ious(lab, n, gt_masks):
    """Matched IoU of every candidate against one reading (0 if unmatched).
    A match requires IoU > 0.5, which makes it unique."""
    out = np.zeros(n, np.float32)
    if n == 0:
        return out
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float32)
    for g in gt_masks:
        inter = np.bincount(lab[g], minlength=n + 1)[1:].astype(np.float32)
        iou = inter / np.maximum(area + g.sum() - inter, 1)
        j = int(np.argmax(iou))
        if iou[j] > 0.5 and iou[j] > out[j]:
            out[j] = iou[j]
    return out


# ---------------------------------------------------------------- feature tables
def _fold_files(paths, split):
    return [paths.prob(split, k) for k in range(C.N_FOLDS) if os.path.exists(paths.prob(split, k))]


def build_oof_table(paths, mask_t, limit=0):
    """Candidates of all out-of-fold training images with their features and
    their matched IoU against every reading. Returns (table, image info)."""
    ann = Annotations(paths.annotations)
    disk = load_disk_cache(paths.disk_cache)
    rows, imgs, t0 = [], {}, time.time()
    for k in range(C.N_FOLDS):
        if not os.path.exists(paths.prob("val", k)):
            continue
        z = np.load(paths.prob("val", k))
        names = [n for n in z.files if n in disk and n in ann.readings_of]
        for name in (names[:limit] if limit else names):
            img = read_gray(os.path.join(paths.train_images, name))
            lab, df = candidate_features(z[name], img, disk[name], mask_t)
            n = len(df)
            gts = [ann.instance_masks(iid) for iid in ann.readings_of[name]]
            R = len(gts)
            iou = (np.stack([match_ious(lab, n, g) for g in gts], 1) if n
                   else np.zeros((0, R), np.float32))
            df["obs"], df["fold"], df["R"] = name, k, R
            df["iou_sum"] = iou.sum(1)            # summed over readings
            df["h"] = (iou > 0).mean(1)           # fraction of readings that match
            df["iou_rows"] = list(iou)
            rows.append(df)
            imgs[name] = dict(fold=k, R=R, n_gt=sum(len(g) for g in gts))
            if len(imgs) % 50 == 0:
                print(f"  OOF features: {len(imgs)} images ({time.time() - t0:.0f}s)", flush=True)
    table = pd.concat(rows, ignore_index=True)
    print(f"OOF candidates: {len(table)} in {len(imgs)} images ({time.time() - t0:.0f}s)")
    return table, imgs


def ensemble_prob(archives, name):
    """Mean of the fold models' uint8 probability maps, rounded to uint8."""
    acc = np.zeros((C.FULL, C.FULL), np.float32)
    for z in archives:
        acc += z[name]
    return np.clip(acc / len(archives) + 0.5, 0, 255).astype(np.uint8)


def build_test_table(paths, mask_t):
    """Candidates and features of the test images from the fold ensemble."""
    disk = load_disk_cache(paths.disk_cache)
    archives = [np.load(f) for f in _fold_files(paths, "test")]
    if not archives:
        raise FileNotFoundError(f"no test_prob_fold*.npz in {paths.work_dir}")
    names = sorted(set(archives[0].files).intersection(*[set(z.files) for z in archives[1:]]))
    rows, t0 = [], time.time()
    for i, name in enumerate(names):
        img = read_gray(os.path.join(paths.test_images, name))
        _, df = candidate_features(ensemble_prob(archives, name), img, disk[name], mask_t)
        df["obs"] = name
        df["cand_id"] = np.arange(1, len(df) + 1)
        rows.append(df)
        if (i + 1) % 60 == 0:
            print(f"  test features: {i + 1}/{len(names)} ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(rows, ignore_index=True), names, archives


# ---------------------------------------------------------------- scorer
def fit_scorer(table, seed=0):
    """Classifier for h (one row per candidate x reading) and regressor for m
    (matched rows only)."""
    X = table[FEATURES].to_numpy(np.float32)
    r_max = int(table["R"].max())
    iou = np.stack(table["iou_rows"].map(lambda a: np.pad(a, (0, r_max - len(a)))).to_list())
    n_read = table["R"].to_numpy()
    rep = np.concatenate([np.repeat(i, r) for i, r in enumerate(n_read)])
    col = np.concatenate([np.arange(r) for r in n_read])
    y_iou = iou[rep, col]
    Xr = X[rep]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=seed)
    clf.fit(Xr, (y_iou > 0).astype(int))
    matched = y_iou > 0
    reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                        min_samples_leaf=40, l2_regularization=1.0,
                                        random_state=seed)
    reg.fit(Xr[matched], y_iou[matched])
    return clf, reg


def score(models, table):
    """(h, m, u = h * m) for every candidate."""
    clf, reg = models
    if len(table) == 0:
        z = np.zeros(0)
        return z, z, z
    X = table[FEATURES].to_numpy(np.float32)
    h = clf.predict_proba(X)[:, 1]
    m = np.clip(reg.predict(X), 0.5, 1.0)
    return h, m, h * m


def cross_fit(table):
    """Out-of-fold scores: candidates of fold k are scored by models fitted on
    the other folds. Adds the columns h_hat, m_hat and u_hat."""
    u, h, m = np.zeros(len(table)), np.zeros(len(table)), np.zeros(len(table))
    for k in sorted(table["fold"].unique()):
        train, test = table["fold"] != k, (table["fold"] == k).to_numpy()
        h[test], m[test], u[test] = score(fit_scorer(table[train]), table[test])
    return table.assign(u_hat=u, h_hat=h, m_hat=m)


# ---------------------------------------------------------------- fast PQ and bootstrap
def per_image_terms(table, selected, imgs):
    """(numerator, denominator) of PQ contributed by each image."""
    g = table.assign(_num=np.where(selected, table["iou_sum"], 0.0),
                     _k=selected.astype(int)).groupby("obs")
    num, k = g["_num"].sum(), g["_k"].sum()
    return {name: (float(num.get(name, 0.0)),
                   0.5 * info["R"] * int(k.get(name, 0)) + 0.5 * info["n_gt"])
            for name, info in imgs.items()}


def pq_from(terms, keys=None):
    keys = keys if keys is not None else list(terms)
    num = sum(terms[k][0] for k in keys)
    den = sum(terms[k][1] for k in keys)
    return num / den if den else 0.0


def paired_bootstrap(terms_a, terms_b, n_boot=2000, seed=0):
    """95 % CI of PQ(b) - PQ(a), resampling images, and P(difference <= 0)."""
    keys = list(terms_a)
    na = np.array([terms_a[k][0] for k in keys]); da = np.array([terms_a[k][1] for k in keys])
    nb = np.array([terms_b[k][0] for k in keys]); db = np.array([terms_b[k][1] for k in keys])
    rng = np.random.default_rng(seed)
    diff = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(keys), len(keys))
        diff[b] = nb[i].sum() / db[i].sum() - na[i].sum() / da[i].sum()
    return np.percentile(diff, [2.5, 97.5]), float((diff <= 0).mean())


# ---------------------------------------------------------------- fit
def fit(paths, model_dir, mask_t=0.3, limit=0):
    """Cross-fitted evaluation on out-of-fold candidates, then the final scorer.

    Candidates of fold k are scored by models trained on the other folds, so
    the reported PQ is honest. The hysteresis-threshold baseline (emit iff max
    probability > seed_t and area >= min_area) is evaluated on the same
    candidates. Writes scorer.pkl and decision_layer_params.json."""
    table, imgs = build_oof_table(paths, mask_t, limit)
    folds = sorted(table["fold"].unique())

    best = (-1.0, None, None)
    for seed_t in [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]:
        for min_area in [150, 300, 500, 800]:
            sel = ((table["maxp"] > np.floor(seed_t * 255) / 255)
                   & (table["area"] >= min_area)).to_numpy()
            pq = pq_from(per_image_terms(table, sel, imgs))
            if pq > best[0]:
                best = (pq, (seed_t, min_area), sel)
    base_pq, (seed_t, min_area), base_sel = best
    print(f"baseline (hysteresis): seed_t={seed_t}, min_area={min_area} -> OOF PQ {base_pq:.4f}")

    table = cross_fit(table)

    taus = np.round(np.arange(0.05, 0.60, 0.01), 3)
    curve = [(t, pq_from(per_image_terms(table, (table["u_hat"] > t).to_numpy(), imgs)))
             for t in taus]
    tau, dl_pq = max(curve, key=lambda c: c[1])
    dl_sel = (table["u_hat"] > tau).to_numpy()
    print(f"decision layer: tau={tau} -> OOF PQ {dl_pq:.4f}   (theory: PQ/2 = {dl_pq / 2:.3f})")

    terms_a = per_image_terms(table, base_sel, imgs)
    terms_b = per_image_terms(table, dl_sel, imgs)
    ci, p_le0 = paired_bootstrap(terms_a, terms_b)
    print(f"delta PQ = {dl_pq - base_pq:+.4f}   paired bootstrap 95% CI "
          f"[{ci[0]:+.4f}, {ci[1]:+.4f}]   P(delta<=0) = {p_le0:.3f}")
    per_fold = {}
    print("per fold (baseline -> decision layer):")
    for k in folds:
        keys = [n for n, info in imgs.items() if info["fold"] == k]
        per_fold[int(k)] = (pq_from(terms_a, keys), pq_from(terms_b, keys))
        print(f"  fold {k}: {per_fold[int(k)][0]:.4f} -> {per_fold[int(k)][1]:.4f}")
    per_image = dl_sel.sum() / len(imgs)
    print(f"instances per image: baseline {base_sel.sum() / len(imgs):.2f} | "
          f"decision layer {per_image:.2f}")

    bins = pd.cut(table["h_hat"], [0, .1, .2, .3, .5, .7, .9, 1.0], include_lowest=True)
    cal = table.groupby(bins, observed=True).agg(n=("h", "size"), predicted=("h_hat", "mean"),
                                                observed=("h", "mean"))
    print("calibration of h (predicted vs observed match rate):\n" + cal.round(3).to_string())

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "scorer.pkl"), "wb") as f:
        pickle.dump(fit_scorer(table), f)         # final scorer: all OOF candidates
    params = dict(
        mask_t=mask_t, tau=float(tau), instances_per_image=float(per_image),
        oof_pq=float(dl_pq), baseline_pq=float(base_pq), baseline_seed_t=seed_t,
        baseline_min_area=min_area, delta_ci95=[float(ci[0]), float(ci[1])],
        per_fold={k: [float(a), float(b)] for k, (a, b) in per_fold.items()},
        tau_curve=[[float(t), float(p)] for t, p in curve],
        n_images=len(imgs), n_candidates=len(table), features=FEATURES,
        sklearn_version=sklearn.__version__)
    with open(os.path.join(model_dir, "decision_layer_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"saved scorer and parameters to {model_dir}/")
    return params


# ---------------------------------------------------------------- predict
def load_scorer(model_dir):
    with open(os.path.join(model_dir, "decision_layer_params.json")) as f:
        params = json.load(f)
    if params["sklearn_version"] != sklearn.__version__:
        print(f"WARNING: scorer was fitted with scikit-learn {params['sklearn_version']}, "
              f"running {sklearn.__version__}; install the pinned version to reproduce exactly")
    with open(os.path.join(model_dir, "scorer.pkl"), "rb") as f:
        return pickle.load(f), params


def predict(paths, model_dir, out_csv, calibrate=True):
    """Score the test candidates and write the submission file.

    The out-of-fold probability maps come from single models, the test maps
    from a 5-model average, whose peaks are lower. This shifts the scorer's
    inputs, so with `calibrate` the threshold is re-chosen such that the mean
    number of emitted instances per test image equals the out-of-fold value.
    No test labels are involved."""
    models, params = load_scorer(model_dir)
    mask_t, tau = params["mask_t"], params["tau"]
    table, names, archives = build_test_table(paths, mask_t)
    table = table.assign(u_hat=score(models, table)[2])
    if calibrate and len(archives) > 1:
        grid = np.round(np.arange(0.02, 0.70, 0.005), 3)
        count = np.array([(table["u_hat"] > t).sum() / len(names) for t in grid])
        tau = float(grid[np.argmin(np.abs(count - params["instances_per_image"]))])
    print(f"tau: {params['tau']} out-of-fold ({params['instances_per_image']:.2f} instances/image)"
          f" -> {tau:.3f} on test ({(table['u_hat'] > tau).sum() / len(names):.2f}/image)")

    disk = load_disk_cache(paths.disk_cache)
    rows = []
    for name in names:
        keep = table[(table["obs"] == name) & (table["u_hat"] > tau)]["cand_id"].to_numpy()
        if len(keep) == 0:
            continue
        valid = disk_grid(disk[name])[2] < C.DISK_FRAC
        lab, _ = label_candidates(ensemble_prob(archives, name), valid, mask_t)
        stem = os.path.splitext(name)[0]
        for j, cand in enumerate(keep, 1):
            counts = mask_util.encode(np.asfortranarray((lab == cand).astype(np.uint8)))["counts"]
            rows.append({"filament_id": f"{stem}_{j}",
                         "segmentation_rle": counts.decode() if isinstance(counts, bytes) else counts})
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    sub.to_csv(out_csv, index=False)
    problems = check_submission(out_csv)
    print(f"wrote {out_csv}: {len(sub)} instances in {len(names)} images | "
          f"format check {'passed' if problems == 0 else f'FAILED ({problems})'}")
    return sub


def decode_submission(path):
    """{image stem: [boolean masks]} from a submission CSV."""
    sub = pd.read_csv(path)
    out = defaultdict(list)
    for fid, counts in zip(sub["filament_id"], sub["segmentation_rle"]):
        m = mask_util.decode({"size": [C.FULL, C.FULL], "counts": counts.encode()})
        out[fid.rsplit("_", 1)[0]].append(m.astype(bool))
    return out


def check_submission(path):
    """Number of format problems: duplicate ids, empty or overlapping masks."""
    problems = int(not pd.read_csv(path)["filament_id"].is_unique)
    for masks in decode_submission(path).values():
        problems += sum(int(m.sum() == 0) for m in masks)
        problems += int(np.sum(masks, axis=0).max() > 1)
    return problems


def compare_submissions(path_a, path_b):
    """Summary of the differences between two submission files."""
    a, b = decode_submission(path_a), decode_submission(path_b)
    n_a, n_b = sum(map(len, a.values())), sum(map(len, b.values()))
    identical, differing_pixels = 0, 0
    for stem in set(a) | set(b):
        ua = np.any(a.get(stem, [np.zeros((C.FULL, C.FULL), bool)]), axis=0)
        ub = np.any(b.get(stem, [np.zeros((C.FULL, C.FULL), bool)]), axis=0)
        differing_pixels += int((ua ^ ub).sum())
        keys_b = {m.tobytes() for m in b.get(stem, [])}
        identical += sum(m.tobytes() in keys_b for m in a.get(stem, []))
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        same_bytes = fa.read() == fb.read()
    return dict(byte_identical=same_bytes, instances_a=n_a, instances_b=n_b,
                identical_instances=identical, differing_pixels=differing_pixels)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fit", "predict"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--work-dir", default="output", help="folder with *_prob_fold*.npz")
    ap.add_argument("--disk-cache", default="data/disk_cache_v2.json")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--mask-t", type=float, default=0.3, help="candidate threshold (fit)")
    ap.add_argument("--limit", type=int, default=0, help="first N images per fold (fit, smoke test)")
    ap.add_argument("--out", default="output/submission.csv")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="use the out-of-fold tau unchanged (predict)")
    args = ap.parse_args()
    paths = C.Paths(args.data_root, work_dir=args.work_dir, disk_cache=args.disk_cache)
    if args.command == "fit":
        fit(paths, args.model_dir, args.mask_t, args.limit)
    else:
        predict(paths, args.model_dir, args.out, calibrate=not args.no_calibrate)


if __name__ == "__main__":
    main()
