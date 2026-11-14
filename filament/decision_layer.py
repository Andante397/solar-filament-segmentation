"""
decision_layer.py —— 轻量版决策层：用学出来的打分器决定输出哪些候选（只用 CPU）

思路
  现在的规则：一个候选（概率图阈值化后的连通区域）只要 最高概率 > seed_t 且 面积 >= min_area 就输出。
  这里改成：对每个候选算约 20 个特征，训练两个梯度提升模型
      h_hat = P(标注员画了它)          （逐"候选 x 标注员"训练的分类器）
      m_hat = E[IoU | 画了]            （只在匹配上的样本上训练的回归器）
  输出条件  u_hat = h_hat * m_hat > tau。
  理论依据：PQ = sum(匹配 IoU) / (TP + FP/2 + FN/2)。多输出一个候选，分母对每个标注员都 +0.5，
  分子期望 +u，所以当且仅当 u > PQ/2 时输出能让 PQ 变高。tau 在 OOF 上搜索（理论值约 PQ*/2）。

交叉拟合
  fold k 的候选由"用其他 4 折候选训练的模型"打分，所以 OOF 上的评估是诚实的。
  测试集用全部 OOF 候选训练的模型打分。

对照
  在完全相同的候选上，用同一套 PQ 计算评估滞后阈值规则（网格搜索 seed_t、min_area），
  并做按图像的配对 bootstrap，给出 ΔPQ 的置信区间。

PQ 的快速计算
  候选互不重叠，IoU>0.5 的匹配是唯一的，所以每个候选对每个标注员是否匹配，与输出了哪些别的候选无关。
  于是对任意选择 S：  分子 = S 中各候选在各读数上的匹配 IoU 之和
                       分母 = 0.5 * sum_i R_i*|S_i| + 0.5 * (标注实例总数)
  可以在几毫秒内评估任意阈值。

用法
  python decision_layer.py --anno <标注JSON> --disk disk_cache_v2.json --prob-dir ~/probs \
      --train-img-dir <train_images> --test-img-dir <test_images>
  第一次运行会把特征缓存到 --cache-dir（默认 dl_cache/），之后重跑只需几十秒。
  --limit 20 试跑；--mask-t 0.2 0.3 同时试两种候选划分。
"""
import os
import json
import time
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

Image.MAX_IMAGE_PIXELS = None
FULL = 2048
MIN_CAND_AREA = 50            # 小于这个面积的连通区域不当候选（基线规则的 min_area 网格从 150 起）
RING = 15                      # 计算"周围亮度"的环宽（px）

FEATURES = ["log_area", "maxp", "meanp", "log_sump", "frac_p05", "frac_p09", "log_core99",
            "elong", "log_length", "width", "bbox_fill", "rho", "abs_y",
            "con_mean", "con_min", "con_ring", "con_diff", "inten_rel",
            "n_cand_img", "log_nn_dist", "area_rank"]


# ---------------------------------------------------------------- 图像与几何
def disk_grid(d):
    yy, xx = np.mgrid[0:FULL, 0:FULL].astype(np.float32)
    rr = np.hypot(xx - d["cx"], yy - d["cy"]) / d["R"]
    return xx, yy, rr


def contrast_map(img, d, rr):
    """临边校正后的对比度 I / I_clv(r) - 1，用 disk 缓存里的径向中位数廓线（与 notebook 相同）"""
    prof = np.asarray(d["prof"], np.float32)
    nb = len(prof)
    idx = np.clip(rr * nb, 0, nb - 1)
    flat = np.interp(idx, np.arange(nb), prof)
    c = img.astype(np.float32) / np.maximum(flat, 1e-3) - 1.0
    return np.where(rr < 0.995, c, 0.0).astype(np.float32)


# ---------------------------------------------------------------- 候选与特征
def candidates_and_features(prob_u8, img, d, mask_t):
    xx, yy, rr = disk_grid(d)
    valid = rr < 0.995
    lab, n = ndi.label((prob_u8 > int(np.floor(mask_t * 255))) & valid)
    if n == 0:
        return lab, pd.DataFrame(columns=FEATURES)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    keep = area >= MIN_CAND_AREA
    keep[0] = False
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    lab = remap[lab]
    n = int(keep.sum())
    if n == 0:
        return lab, pd.DataFrame(columns=FEATURES)
    L = lab.ravel()
    on = L > 0
    Li = L[on]
    idx = np.arange(1, n + 1)
    p = prob_u8.ravel()[on].astype(np.float32) / 255.0

    def bc(w=None):
        return np.bincount(Li, weights=w, minlength=n + 1)[1:]

    A = bc()
    sump = bc(p)
    mx = ndi.maximum(prob_u8, lab, idx).astype(np.float32) / 255.0
    f05 = bc((p > 0.5).astype(np.float32)) / A
    f09 = bc((p > 0.9).astype(np.float32)) / A
    core = bc((p > 0.99).astype(np.float32))

    # 形状：二阶矩
    x = xx.ravel()[on]; y = yy.ravel()[on]
    mxx, myy = bc(x) / A, bc(y) / A
    sxx = bc(x * x) / A - mxx ** 2
    syy = bc(y * y) / A - myy ** 2
    sxy = bc(x * y) / A - mxx * myy
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-3)
    length = 4 * np.sqrt(np.maximum(l1, 1e-3))          # 均匀细条的长度 ≈ sqrt(12*l1)，量级一致即可
    sl = ndi.find_objects(lab)
    bbox_area = np.array([(s[0].stop - s[0].start) * (s[1].stop - s[1].start) for s in sl], np.float32)

    # 位置
    rho = np.hypot(mxx - d["cx"], myy - d["cy"]) / d["R"]
    abs_y = np.abs(myy - d["cy"]) / d["R"]               # 图像北在上，|Y| 是纬度的近似

    # 图像：对比度、环形背景
    con = contrast_map(img, d, rr)
    cv = con.ravel()[on]
    con_mean = bc(cv) / A
    con_min = ndi.minimum(con, lab, idx).astype(np.float32)
    dil = ndi.grey_dilation(lab, size=(2 * RING + 1, 2 * RING + 1))
    ring = (dil > 0) & (lab == 0) & valid
    rl = dil[ring]
    ring_n = np.bincount(rl, minlength=n + 1)[1:]
    ring_s = np.bincount(rl, weights=con[ring], minlength=n + 1)[1:]
    con_ring = np.where(ring_n > 0, ring_s / np.maximum(ring_n, 1), 0.0)
    disk_med = float(np.median(img[valid])) if valid.any() else 1.0
    inten_rel = (bc(img.ravel()[on].astype(np.float32)) / A) / max(disk_med, 1.0)

    # 上下文
    cxy = np.c_[mxx, myy]
    if n > 1:
        dd = np.hypot(cxy[:, None, 0] - cxy[None, :, 0], cxy[:, None, 1] - cxy[None, :, 1])
        np.fill_diagonal(dd, np.inf)
        nn = dd.min(1)
    else:
        nn = np.full(n, 3000.0)
    rank = (np.argsort(np.argsort(-A)) + 1) / n

    df = pd.DataFrame(dict(
        log_area=np.log(A), maxp=mx, meanp=sump / A, log_sump=np.log1p(sump), frac_p05=f05, frac_p09=f09,
        log_core99=np.log1p(core), elong=np.sqrt(l1 / l2), log_length=np.log(length), width=A / length,
        bbox_fill=A / np.maximum(bbox_area, 1), rho=rho, abs_y=abs_y,
        con_mean=con_mean, con_min=con_min, con_ring=con_ring, con_diff=con_mean - con_ring,
        inten_rel=inten_rel, n_cand_img=np.full(n, n, np.float32), log_nn_dist=np.log1p(nn), area_rank=rank))
    df["area"] = A
    return lab, df


def match_ious(lab, n, gt_masks):
    """每个候选在这一个读数里的匹配 IoU（>0.5 才算，否则 0）。候选互不重叠，匹配唯一。"""
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


# ---------------------------------------------------------------- 构建 OOF / 测试特征表（带缓存）
def build_oof(args, coco, readings, DISK, mask_t):
    path = os.path.join(args.cache_dir, f"oof_mt{mask_t}_lim{args.limit}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    rows, imgs, t0 = [], {}, time.time()
    for k in range(5):
        f = os.path.join(args.prob_dir, f"val_prob_fold{k}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        names = [n for n in z.files if n in DISK and n in readings]
        if args.limit:
            names = names[:args.limit]
        for fn in names:
            img = np.asarray(Image.open(os.path.join(args.train_img_dir, fn)).convert("L"))
            lab, df = candidates_and_features(z[fn], img, DISK[fn], mask_t)
            n = len(df)
            gts = []
            for iid in readings[fn]:
                gts.append([coco.annToMask(a).astype(bool) for a in coco.loadAnns(coco.getAnnIds(imgIds=[iid]))])
            R = len(gts)
            I = np.stack([match_ious(lab, n, g) for g in gts], 1) if n else np.zeros((0, R), np.float32)
            df["obs"], df["fold"], df["R"] = fn, k, R
            df["iou_sum"] = I.sum(1)                       # 各读数匹配 IoU 之和
            df["h"] = (I > 0).mean(1)                      # 被画出的比例
            df["iou_rows"] = list(I)
            rows.append(df)
            imgs[fn] = dict(fold=k, R=R, n_gt=sum(len(g) for g in gts))
            if len(imgs) % 25 == 0:
                print(f"  OOF 特征 {len(imgs)} 张 ({time.time()-t0:.0f}s)", flush=True)
    data = (pd.concat(rows, ignore_index=True), imgs)
    os.makedirs(args.cache_dir, exist_ok=True)
    pickle.dump(data, open(path, "wb"))
    print(f"OOF 候选 {len(data[0])} 个 / {len(imgs)} 张图，{time.time()-t0:.0f}s，已缓存 {path}")
    return data


def ensemble_prob(zs, fn):
    acc = np.zeros((FULL, FULL), np.float32)
    for z in zs:
        acc += z[fn]
    return np.clip(acc / len(zs) + 0.5, 0, 255).astype(np.uint8)


def build_test(args, DISK, mask_t):
    path = os.path.join(args.cache_dir, f"test_mt{mask_t}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    zs = [np.load(os.path.join(args.prob_dir, f"test_prob_fold{k}.npz")) for k in range(5)
          if os.path.exists(os.path.join(args.prob_dir, f"test_prob_fold{k}.npz"))]
    names = sorted(set(zs[0].files).intersection(*[set(z.files) for z in zs[1:]]))
    rows, t0 = [], time.time()
    for i, fn in enumerate(names):
        img = np.asarray(Image.open(os.path.join(args.test_img_dir, fn)).convert("L"))
        prob = ensemble_prob(zs, fn)
        lab, df = candidates_and_features(prob, img, DISK[fn], mask_t)
        df["obs"] = fn
        df["cand_id"] = np.arange(1, len(df) + 1)
        rows.append(df)
        if (i + 1) % 30 == 0:
            print(f"  测试特征 {i+1}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    data = (pd.concat(rows, ignore_index=True), names, len(zs))
    pickle.dump(data, open(path, "wb"))
    return data


# ---------------------------------------------------------------- 打分器
def fit_scorer(df, seed=0):
    """h 分类器：每个 (候选, 读数) 一行；m 回归器：只用匹配上的行"""
    X = df[FEATURES].to_numpy(np.float32)
    Rmax = int(df["R"].max())
    I = np.stack(df["iou_rows"].map(lambda a: np.pad(a, (0, Rmax - len(a)))).to_list())
    Rs = df["R"].to_numpy()
    rep = np.concatenate([np.repeat(i, r) for i, r in enumerate(Rs)])
    col = np.concatenate([np.arange(r) for r in Rs])
    y_iou = I[rep, col]
    Xr = X[rep]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
    clf.fit(Xr, (y_iou > 0).astype(int))
    m = y_iou > 0
    reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                        min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
    reg.fit(Xr[m], y_iou[m])
    return clf, reg


def score(models, df):
    clf, reg = models
    if len(df) == 0:
        z = np.zeros(0)
        return z, z, z
    X = df[FEATURES].to_numpy(np.float32)
    h = clf.predict_proba(X)[:, 1]
    mm = np.clip(reg.predict(X), 0.5, 1.0)
    return h, mm, h * mm


# ---------------------------------------------------------------- PQ（快速）与配对 bootstrap
def per_image(df, sel, imgs):
    """每张图的 (分子, 分母)"""
    g = df.assign(_num=np.where(sel, df["iou_sum"], 0.0), _k=sel.astype(int)).groupby("obs")
    num = g["_num"].sum(); k = g["_k"].sum()
    out = {}
    for fn, info in imgs.items():
        kk = int(k.get(fn, 0))
        out[fn] = (float(num.get(fn, 0.0)), 0.5 * info["R"] * kk + 0.5 * info["n_gt"])
    return out


def pq_from(pi, keys=None):
    keys = keys if keys is not None else list(pi)
    n = sum(pi[k][0] for k in keys); d = sum(pi[k][1] for k in keys)
    return n / d if d else 0.0


def paired_boot(pa, pb, B=2000, seed=0):
    keys = np.array(list(pa))
    na = np.array([pa[k][0] for k in keys]); da = np.array([pa[k][1] for k in keys])
    nb = np.array([pb[k][0] for k in keys]); db = np.array([pb[k][1] for k in keys])
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, len(keys), len(keys))
        out[b] = nb[i].sum() / db[i].sum() - na[i].sum() / da[i].sum()
    return np.percentile(out, [2.5, 97.5]), float((out <= 0).mean())


# ---------------------------------------------------------------- 主流程
def run_one(args, coco, readings, DISK, mask_t):
    print("\n" + "=" * 76 + f"\n候选划分 mask_t = {mask_t}\n" + "=" * 76)
    df, imgs = build_oof(args, coco, readings, DISK, mask_t)
    folds = sorted(df["fold"].unique())

    # 基线：滞后阈值规则，在同一批候选上网格搜索
    best_base = (-1, None)
    for st in [0.90, 0.95, 0.97, 0.98, 0.99, 0.995]:
        for ma in [150, 300, 500, 800]:
            sel = ((df["maxp"] > np.floor(st * 255) / 255) & (df["area"] >= ma)).to_numpy()
            pq = pq_from(per_image(df, sel, imgs))
            if pq > best_base[0]:
                best_base = (pq, (st, ma), sel)
    base_pq, (st, ma), base_sel = best_base
    print(f"基线（滞后阈值）: seed_t={st}, min_area={ma} -> OOF PQ {base_pq:.4f}")

    # 交叉拟合
    u = np.zeros(len(df)); h = np.zeros(len(df)); mm = np.zeros(len(df))
    for k in folds:
        tr, te = df["fold"] != k, df["fold"] == k
        models = fit_scorer(df[tr])
        h[te.to_numpy()], mm[te.to_numpy()], u[te.to_numpy()] = score(models, df[te])
    df = df.assign(u_hat=u, h_hat=h, m_hat=mm)

    taus = np.round(np.arange(0.05, 0.60, 0.01), 3)
    curve = [(t, pq_from(per_image(df, (df["u_hat"] > t).to_numpy(), imgs))) for t in taus]
    tau, dl_pq = max(curve, key=lambda c: c[1])
    dl_sel = (df["u_hat"] > tau).to_numpy()
    print(f"决策层: tau={tau} -> OOF PQ {dl_pq:.4f}   （理论 tau ≈ PQ/2 = {dl_pq/2:.3f}）")

    pa, pb = per_image(df, base_sel, imgs), per_image(df, dl_sel, imgs)
    ci, p_le0 = paired_boot(pa, pb)
    print(f"ΔPQ = {dl_pq - base_pq:+.4f}   配对 bootstrap 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]   P(Δ<=0) = {p_le0:.3f}")
    print("各折 PQ（基线 -> 决策层）:")
    for k in folds:
        ks = [fn for fn, inf in imgs.items() if inf["fold"] == k]
        print(f"  fold {k}: {pq_from(pa, ks):.4f} -> {pq_from(pb, ks):.4f}")
    print(f"平均每图输出实例数: 基线 {base_sel.sum()/len(imgs):.2f} | 决策层 {dl_sel.sum()/len(imgs):.2f}")
    print("tau 曲线（每 0.05）: " + "  ".join(f"{t:.2f}:{p:.4f}" for t, p in curve if abs(t * 20 - round(t * 20)) < 1e-6))

    # 校准检查：按 h_hat 分箱，预测 vs 实际
    bins = pd.cut(df["h_hat"], [0, .1, .2, .3, .5, .7, .9, 1.0], include_lowest=True)
    cal = df.groupby(bins, observed=True).agg(n=("h", "size"), h_pred=("h_hat", "mean"), h_true=("h", "mean"))
    print("h_hat 校准（预测 vs 实际被画出的比例）:\n" + cal.round(3).to_string())
    return dict(mask_t=mask_t, df=df, imgs=imgs, base=(base_pq, st, ma), dl=(dl_pq, tau))


def write_submission(args, DISK, res, mode):
    mask_t, (dl_pq, tau) = res["mask_t"], res["dl"]
    df = res["df"]
    models = fit_scorer(df)                                 # 全部 OOF 候选训练
    tdf, names, n_models = build_test(args, DISK, mask_t)
    _, _, ut = score(models, tdf)
    tdf = tdf.assign(u_hat=ut)
    target = (df["u_hat"] > tau).sum() / len(res["imgs"])
    tau_use = tau
    if mode == "match" and n_models > 1:
        # 集成概率图的峰值被平均压低，打分器的输入分布会偏移；按"每图实例数与 OOF 一致"重新选 tau
        grid = np.round(np.arange(0.02, 0.70, 0.005), 3)
        cnt = np.array([(tdf["u_hat"] > t).sum() / len(names) for t in grid])
        tau_use = float(grid[np.argmin(np.abs(cnt - target))])
    print(f"\n[{mode}] OOF tau={tau}（每图 {target:.2f} 个）-> 测试集用 tau={tau_use:.3f}"
          f"（每图 {(tdf['u_hat'] > tau_use).sum()/len(names):.2f} 个）")

    zs = [np.load(os.path.join(args.prob_dir, f"test_prob_fold{k}.npz")) for k in range(5)
          if os.path.exists(os.path.join(args.prob_dir, f"test_prob_fold{k}.npz"))]
    out_rows = []
    for fn in names:
        keep = tdf[(tdf["obs"] == fn) & (tdf["u_hat"] > tau_use)]["cand_id"].to_numpy()
        if len(keep) == 0:
            continue
        d = DISK[fn]
        yy, xx = np.ogrid[0:FULL, 0:FULL]
        valid = (xx - d["cx"]) ** 2 + (yy - d["cy"]) ** 2 < (0.995 * d["R"]) ** 2
        prob = ensemble_prob(zs, fn)
        lab, n = ndi.label((prob > int(np.floor(mask_t * 255))) & valid)
        area = np.bincount(lab.ravel(), minlength=n + 1)
        k_ = area >= MIN_CAND_AREA; k_[0] = False
        remap = np.zeros(n + 1, np.int32); remap[k_] = np.arange(1, k_.sum() + 1)
        lab = remap[lab]                                    # 与特征计算时完全相同的编号
        stem = os.path.splitext(fn)[0]
        for j, cid in enumerate(keep, 1):
            c = mask_util.encode(np.asfortranarray((lab == cid).astype(np.uint8)))["counts"]
            out_rows.append({"filament_id": f"{stem}_{j}", "segmentation_rle": c.decode() if isinstance(c, bytes) else c})
    path = os.path.join(args.out_dir, f"submission_dl_{mode}.csv")
    sub = pd.DataFrame(out_rows, columns=["filament_id", "segmentation_rle"])
    sub.to_csv(path, index=False)
    # 自检
    chk = pd.read_csv(path)
    assert chk.filament_id.is_unique
    bad, per = 0, defaultdict(list)
    for fid, c in zip(chk.filament_id, chk.segmentation_rle):
        per[fid.rsplit("_", 1)[0]].append(c)
    for _, cs in per.items():
        acc = np.zeros((FULL, FULL), np.uint8)
        for c in cs:
            m = mask_util.decode({"size": [FULL, FULL], "counts": c.encode()})
            bad += int(m.sum() == 0); acc += m
        bad += int(acc.max() > 1)
    print(f"已写入 {path}: {len(sub)} 个实例 / {len(names)} 张图 | 自检 {'通过' if bad == 0 else f'{bad} 处异常'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", required=True)
    ap.add_argument("--disk", required=True)
    ap.add_argument("--prob-dir", required=True)
    ap.add_argument("--train-img-dir", required=True)
    ap.add_argument("--test-img-dir", default=None)
    ap.add_argument("--mask-t", type=float, nargs="+", default=[0.3])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-dir", default="dl_cache")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--no-submission", action="store_true")
    args = ap.parse_args()

    DISK = json.load(open(args.disk))
    coco = COCO(args.anno)
    readings = defaultdict(list)
    for iid in coco.getImgIds():
        readings[coco.loadImgs([iid])[0]["file_name"]].append(iid)

    results = [run_one(args, coco, readings, DISK, mt) for mt in args.mask_t]
    best = max(results, key=lambda r: r["dl"][0])
    print("\n" + "=" * 76)
    for r in results:
        print(f"mask_t={r['mask_t']}: 基线 {r['base'][0]:.4f} | 决策层 {r['dl'][0]:.4f}")
    print(f"采用 mask_t={best['mask_t']}")
    json.dump(dict(mask_t=best["mask_t"], base_pq=best["base"][0], base_seed_t=best["base"][1],
                   base_min_area=best["base"][2], dl_pq=best["dl"][0], tau=best["dl"][1]),
              open(os.path.join(args.out_dir, "decision_layer_params.json"), "w"), indent=2)

    if not args.no_submission and args.test_img_dir:
        for mode in ("match", "same"):
            write_submission(args, DISK, best, mode)


if __name__ == "__main__":
    main()
