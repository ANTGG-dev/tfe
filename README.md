# TFE for floating debris detection

Code, frozen experiment plans, and statistical evidence for the manuscript:
**Evaluating TFE in YOLO11 P2 Variants for Floating Debris Detection: Performance by Object Size and Computational Cost**.

Repository: https://github.com/ANTGG-dev/tfe

Clone this repository, then follow the installation and data preparation steps below. Training is not started automatically.

## Scope
150 main training jobs (30 configurations, five seeds), 75 grouped-CV jobs,
10 physical-batch-8 sensitivity jobs; 20 RiSID evaluations reuse source weights.
The FloW follow-up adds 20 supervised runs (four architectures, five seeds);
see [FloW reproduction](FLOW.md) for the frozen split, best checkpoints, and calculations.
The exact custom Ultralytics 8.4.21 source includes TinyTFEStage/TinySCA,
ADown/RepC3, parsers and architecture YAMLs. Upstream license/notices are
preserved in cuerposAgua/ultralytics_fork/LICENSE (AGPL-3.0).
The source snapshot includes upstream features not evaluated in the paper;
plans/training_jobs.json defines the original scope; evidence/flow/frozen_plan.json
defines the additional FloW scope.

## Environment and installation
Recorded environment: Windows, Python 3.12, PyTorch 2.10.0+cu128,
RTX 5080, local editable Ultralytics 8.4.21. Consult environment.json and
requirements-recorded.txt for the actual versions. In a new virtual environment:
```
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-recorded.txt
python -m pip install --no-deps -e cuerposAgua/ultralytics_fork
python prepare_portable.py --verify
```
The original environment snapshot is provenance, not a promise that every
recorded package remains available from a registry. No packages are installed
or downloaded automatically by the verification script.

## Data and checkpoints
Dataset images and original-experiment checkpoints are not embedded. The 20
FloW best.pt checkpoints are included under evidence/flow/jobs; intermediate
and last-epoch checkpoints are excluded. Obtain
River-Pollution-Master v5 (lumsworkspace, Roboflow Universe, CC BY 4.0) and
RiSID v2 (https://doi.org/10.5281/zenodo.16927238) from their cited sources.
Place the corrected local export at revision/prepared/dataset/{train,valid,test}/
images and labels. The 988-record manifest provides exact names, hashes,
annotations and source-name families: 828/120/40 images; DJI_0185 is excluded
from train. It does not establish flight independence. CV split lists keep
source-name families together; they do not include final test images.
The main manifest retains original acquisition paths as ROOT placeholders;
the preparation script writes portable paths using the corrected export.
For COCO jobs obtain the named official Ultralytics checkpoints, place them
in revision/models, and verify hashes in pretrained_weights_required.json.
Source-trained best.pt checkpoints must be supplied or trained before external
inference. See revision/risid/plan.json for the exact mapping; RiSID metadata,
images and GT must be prepared as described in the manuscript, not replaced
by the local two-class labels.

## Running a selected training job (explicit opt-in)
```
python prepare_portable.py --materialize
python revision/train_worker.py --job portable_jobs/0000.json
```
Materialization writes all 235 job JSONs and data YAMLs; it does NOT train.
Inspect the selected job before executing it. Training requires CUDA.
The worker resumes existing last.pt and validates the final epoch; preserve
the supplied settings and compare only the reported scope. It never silently
reduces batch size. The original runner/evaluation/profile scripts are included
for inspection and use with the materialized plans; their older plan-building
entry points are not the frozen scope and must not be used to redefine it.

## Statistical evidence
tables contains S1-S15. S13 adds fixed-confidence precision/recall intervals
for all main jobs and batch8 (160 fitted runs). evidence/fixed_pr contains
per-image TP/FP/GT sufficient statistics, bootstrap draws and verification.
Main uncertainty uses 1000 shared source-family resamples, plus image-level
sensitivity, averaging the same five fitted seeds in every draw. Seeds are
not resampled. Intervals are pointwise, conditional, and not equivalence tests.
Undefined recall for zero GT is omitted from percentile estimation and its
valid-draw count is recorded; precision with no detections is zero by the
existing evaluation convention. The class label macro denotes pooled P/R
counts, whereas AP macro averages classes. RiSID reports seed SD, not
independent-video confidence intervals. S14-S15 audit retained logs only.

## Integrity
SHA256SUMS.json identifies the distributed source and evidence files, excluding itself and Git metadata. Verification
checks those hashes. No training, inference or network call is needed.
