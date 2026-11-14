"""COCO annotations, GONG station codes and the cross-validation folds.

Needs no deep-learning framework, so post-processing can run without PyTorch.
"""
import os
import random
from collections import defaultdict

from pycocotools import mask as mask_util
from pycocotools.coco import COCO

from . import config as C


class Annotations:
    """COCO annotations grouped by observation.

    The file has more `image_id`s than image files: an observation labelled by
    several annotators appears once per annotator ("reading"). Cross-validation
    therefore has to be grouped by *file name*, otherwise readings of the same
    image leak between training and validation.
    """

    def __init__(self, path):
        self.coco = COCO(path)
        self.readings = []                       # one entry per annotator reading
        self.readings_of = defaultdict(list)     # file name -> [image_id]
        for iid in self.coco.getImgIds():
            name = self.coco.loadImgs([iid])[0]["file_name"]
            self.readings.append({"image_id": iid, "file": name})
            self.readings_of[name].append(iid)
        self.observations = sorted(self.readings_of)

    def instances(self, image_id):
        return self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))

    def instance_masks(self, image_id):
        return [self.coco.annToMask(a).astype(bool) for a in self.instances(image_id)]

    def union_rle(self, image_id):
        """All filaments of one reading merged into a single RLE (or None)."""
        rles = []
        for a in self.instances(image_id):
            rle = self.coco.annToRLE(a)
            if isinstance(rle["counts"], list):
                rle = mask_util.frPyObjects(rle, rle["size"][0], rle["size"][1])
            rles.append(rle)
        return mask_util.merge(rles) if rles else None

    def centres(self, image_id):
        """Bounding-box centres of the filaments of one reading."""
        return [(a["bbox"][0] + a["bbox"][2] / 2.0, a["bbox"][1] + a["bbox"][3] / 2.0)
                for a in self.instances(image_id)]


def site_of(name):
    """GONG station code from a file name, e.g. 20110120105534Ch.jpeg -> 'Ch'."""
    stem = os.path.splitext(name)[0]
    letters = "".join(ch for ch in stem if ch.isalpha())
    return letters if letters else "??"


def assign_folds(observations, n_folds=C.N_FOLDS, seed=C.SEED):
    """Fold of every observation: grouped by file, stratified by station."""
    by_site = defaultdict(list)
    for o in observations:
        by_site[site_of(o)].append(o)
    rng = random.Random(seed)
    fold = {}
    for _, obs in sorted(by_site.items()):
        obs = sorted(obs)
        rng.shuffle(obs)
        for i, o in enumerate(obs):
            fold[o] = i % n_folds
    return fold
