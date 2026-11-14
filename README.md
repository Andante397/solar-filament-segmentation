# Solar filament segmentation in GONG H-alpha images

Solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData Cup). Instance segmentation of solar filaments in full-disk H-alpha
observations, evaluated with Panoptic Quality (PQ) against multiple annotators.

| | PQ |
|---|---|
| Pairwise agreement between human annotators (296 multi-annotator images) | 0.340 |
| Hysteresis-threshold baseline, 5-fold out-of-fold | 0.432 |
| **This solution, 5-fold out-of-fold** (95% CI 0.431–0.449) | **0.440** |
| This solution, public leaderboard | 0.38 |

## Method

1. **Disk geometry** (`filament/disk.py`). The limb is located as the sharp intensity rise into
   the thin bright ring at the disk edge: Hough circle transform for a coarse fit, then a robust
   circle fit to 720 edge points. Radius on all 887 images: 898.3 ± 0.7 px. The radial median
   intensity profile of each image is used to remove limb darkening.
2. **Segmentation network** (`filament/model.py`, `filament/train.py`). U-Net with a ResNet-34
   encoder and a decoder that runs to full resolution. Input planes: limb-corrected contrast,
   raw intensity, normalised radius r/R. Trained on 512-px tiles at native resolution with one
   sample per annotator reading and a BCE-dominated loss, so the output estimates the
   probability that a randomly chosen annotator marks a pixel. 5-fold cross-validation grouped
   by image and stratified by GONG station.
3. **Inference** (`filament/infer.py`). Sliding window (stride 256) with raised-cosine blending
   and 8-fold dihedral test-time augmentation; the five fold models are averaged.
4. **Decision layer** (`filament/decision_layer.py`). Connected regions with probability > 0.3
   are candidates. Two gradient-boosted models estimate, from 21 features per candidate, the
   probability *h* that an annotator drew a matching filament and the expected IoU *m* of the
   match. A candidate is emitted iff *h·m* > τ. From PQ = Σ IoU / (TP + ½FP + ½FN) the optimal
   threshold is τ = PQ/2, which the out-of-fold search confirms.

## Reproducing the submission

```bash
git clone https://github.com/Andante397/solar-filament-segmentation.git && cd solar-filament-segmentation
pip install -r requirements.txt
jupyter notebook reproduce_submission.ipynb     # set DATA_ROOT in the first code cell
```

The notebook loads the trained weights from `weights/` and the fitted candidate scorer from
`models/`, predicts the 180 test images and writes `output/submission.csv` (about one hour on
a single T4 GPU). Its last cell compares the result with the submitted file,
`submission/submission_dl_match.csv`. GPU convolutions are not bit-reproducible across hardware
and library versions, so a few boundary pixels may differ.

On Kaggle: attach the competition data, clone the repository into `/kaggle/working`, change
into it and run the notebook; the default `DATA_ROOT` is the Kaggle path.

## Repository layout

| path | content |
|---|---|
| `reproduce_submission.ipynb` | end-to-end pipeline from weights to submission file |
| `filament/` | source code (see Method) |
| `weights/unet_fold{0..4}.pt` | trained networks, one per fold |
| `models/scorer.pkl`, `models/decision_layer_params.json` | fitted decision layer and its out-of-fold evaluation |
| `data/disk_cache_v2.json` | disk centre, radius and limb profile of every competition image |
| `submission/submission_dl_match.csv` | the submitted predictions |
| `requirements.txt` | package versions of the environment that produced the submission |
| `analysis/diagnostics.py` | out-of-fold diagnostics and the figures of the technical report (`analysis/out/`) |

## Retraining (optional)

```bash
# network: several GPU hours per fold; results vary slightly between runs
for k in 0 1 2 3 4; do python -m filament.train --data-root $DATA --fold $k; done
# out-of-fold and test probability maps
python -m filament.infer --data-root $DATA --weights-dir weights_retrained
# decision layer: cross-fitted evaluation, then the final scorer
python -m filament.decision_layer fit --data-root $DATA
python -m filament.decision_layer predict --data-root $DATA
```

## Data and external resources

- Only the competition data (MAGFiLO annotations on GONG H-alpha images) were used for training.
  No external observations and no labels of the test images were used.
- The encoder is initialised from the ImageNet weights distributed with torchvision.
- When the test set is predicted, the decision threshold is re-chosen so that the mean number
  of emitted instances per test image equals the out-of-fold value (the ensemble average has
  lower peaks than the single-model out-of-fold maps). This uses test images but no test labels.
- The decision layer follows the public Kaggle notebook *"Deciding under disagreement — a
  PQ-optimal detect-then-refine cascade for solar filaments (filseg v4)"* shared in this
  competition: the rule *h·m* > PQ/2, the split into a match probability and a conditional IoU,
  and the cross-fitted estimation are taken from it. That solution selects candidates from a
  multi-threshold component tree and adds a detector and a refinement stage; this repository uses
  a single threshold with connected components, its own feature set and the calibration step
  described above.

## Acknowledgements

This work uses the MAGFiLO annotations and GONG H-alpha data obtained by the NSO Integrated
Synoptic Program. SDO/HMI magnetograms (courtesy of NASA/SDO and the HMI science team) were used
for analysis only. Parts of the source code were written and refactored with the assistance of
Anthropic's Claude; all results were run and checked by the author.

## License

MIT, see `LICENSE`.
