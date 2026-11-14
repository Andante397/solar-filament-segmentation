"""Paths and hyper-parameters shared by training, inference and post-processing."""
import os
from dataclasses import dataclass

# ---------------------------------------------------------------- geometry
FULL = 2048            # native image size; images are never resampled
TILE = 512             # tile size for training and sliding-window inference
STRIDE = 256           # sliding-window stride (50 % overlap)
NBINS = 256            # radial shells of the limb-darkening profile
LIMB_STEP = 2          # pixel subsampling when estimating the radial profile
DISK_FRAC = 0.995      # pixels with r < DISK_FRAC * R count as on-disk
EXPECTED_RADIUS = 898  # px; GONG images are rescaled to a common solar radius
RADIUS_TOL = 8         # px; a larger deviation means the disk fit failed

# ---------------------------------------------------------------- training
N_FOLDS = 5
SEED = 42              # fold assignment; the training seed of fold k is SEED + k
EPOCHS = 40
STEPS_PER_EPOCH = 400
TILES_PER_ITEM = 8     # tiles drawn from one image per step (= batch size)
LR = 3e-4
WEIGHT_DECAY = 1e-4
POS_TILE_FRAC = 0.75   # fraction of tiles centred on an annotated filament
DICE_WEIGHT = 0.3

# ---------------------------------------------------------------- inference
TTA = 8                # 1 = none, 4 = flips, 8 = full dihedral group

IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


@dataclass
class Paths:
    """File locations. `data_root` is the unpacked competition folder
    (the one containing `train/` and `test/`)."""
    data_root: str
    work_dir: str = "output"       # probability maps, submission
    cache_dir: str = "cache"       # decoded images as memmap (large, disposable)
    weights_dir: str = "weights"   # unet_fold{k}.pt
    disk_cache: str = "data/disk_cache_v2.json"

    @property
    def train_images(self):
        return os.path.join(self.data_root, "train", "train_images")

    @property
    def test_images(self):
        return os.path.join(self.data_root, "test", "test_images")

    @property
    def annotations(self):
        return os.path.join(self.data_root, "train",
                            "MAGFiLO_1.0_Annotations_kaggle2026_train.json")

    def image_dir(self, split):
        return {"train": self.train_images, "test": self.test_images}[split]

    def weights(self, fold):
        return os.path.join(self.weights_dir, f"unet_fold{fold}.pt")

    def prob(self, split, fold):
        """split is 'val' (out-of-fold training images) or 'test'."""
        return os.path.join(self.work_dir, f"{split}_prob_fold{fold}.npz")

    def makedirs(self):
        for d in (self.work_dir, self.cache_dir, os.path.dirname(self.disk_cache) or "."):
            os.makedirs(d, exist_ok=True)


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))
