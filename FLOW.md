# FloW supervised follow-up

This package adds the **20 completed FloW training runs** (four architectures,
seeds 0–4) to the original experiments. `evidence/flow/jobs` contains each
validation-selected `best.pt`, 150-epoch training curve, configuration, actual
training record, completion marker, test predictions, ground truth, metrics,
and shared image-bootstrap draws. Intermediate checkpoints, `last.pt`, logs,
dataset images, virtual environments, and timing-pilot weights are excluded.
Published evidence is kept separate from new reproduction outputs.

## Dataset and protocol

Obtain **FloW_IMG.zip** from the FloW authors:
https://orca-tech.cn/datasets/FloW/FloW-Img

Cheng et al., *FloW: A Dataset and Benchmark for Floating Waste Detection in
Inland Waters*, ICCV 2021, https://doi.org/10.1109/ICCV48922.2021.01077.

Place the archive at `revision/flow/raw/FloW_IMG.zip`.
Expected SHA-256:
`d8fe0b824935a0ae3a5d1252f0aea5e23600e38059de89b12b53fe07afd18e30`.

The official split is 1200 training / 800 test images. We derived 240 validation
images exclusively from the training pool with seed 20261028, leaving **960 /
240 / 800** images and **2638 / 610 / 2024** bottle boxes. The exact released XML
count is 5272, versus 5271 in the paper. No boxes were dropped or clipped.
Exact duplicate hashes do not cross splits; sequence/site independence is not
established. This is a surface-vehicle, single-class dataset, not an aerial
two-class benchmark. `evidence/flow/data` preserves the audited split and hashes.

All runs used scratch initialization, 150 epochs without early stopping, 640-pixel
input, batch 4, nominal batch 16, AdamW, the original augmentation settings,
four workers, and FP32. FP32 was fixed before final training after an AMP pilot
produced a non-finite validation classification loss. `p2_adown_tfe_off` means
**TFE present, TinySCA off**, not TFE absent. Validation selected checkpoints;
all 20 trainings preceded test evaluation. Full settings are in the frozen plan.

## Reproduce

Install the dependencies and editable fork as described in the main README.
Use the portable entry point below, rather than the historical runner's `freeze`
entry point. No training runs unless explicitly requested.

```sh
# No images or GPU needed: check distributed files and recompute the statistics.
python prepare_portable.py --verify
python reproduce_flow.py --verify-results

# After obtaining the original ZIP: reconstruct and verify the frozen split.
python reproduce_flow.py --prepare

# Re-evaluate the 20 published best checkpoints; no training.
python reproduce_flow.py --evaluate-best

# Train all 20, then evaluate and run paired image bootstrap.
python reproduce_flow.py --run
```

On Windows, `iniciar_flow_20.cmd` prepares the data and executes the last command.
It uses `revision/.venv/Scripts/python.exe` when present, otherwise the activated
environment's `python`. Completed reproduction jobs are skipped; interrupted
training resumes from its local `last.pt`. Outputs go to `runs/flow/retrained`
or `runs/flow/published_best`, never into the published evidence directory.
The historical scripts are included for provenance. Source JSON paths use
`${ROOT}` placeholders; the portable entry point expands them. Binary checkpoints
retain their original training metadata and are supplied unchanged.

## Results and uncertainty

| Model | Test AP50–95 mean ± sample SD (%) |
|---|---:|
| YOLO11n | 33.21 ± 0.33 |
| P2 | 34.36 ± 0.85 |
| P2–ADown | 33.27 ± 0.70 |
| P2–ADown–TFE, TinySCA off | 34.53 ± 0.62 |

Three planned contrasts (percentage points), with Bonferroni 98.333% intervals:
P2 − baseline **1.15 [0.67, 1.69]**; ADown added to P2 **−1.09 [−1.60, −0.57]**;
TFE added to P2–ADown **1.26 [0.64, 1.87]**. The 1000 shared image resamples
average five fitted seeds; they condition on those models and do not include
new-training uncertainty. Scene dependence may make the intervals optimistic.
The 0.17-point TFE–P2 mean difference was not a planned inferential contrast.
`summary.json`, `results_per_seed.csv`, and `paired_image_contrasts.json` contain
unrounded results; the verification command recomputes means, SD, and intervals.
