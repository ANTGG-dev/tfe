# FloW-Img supervised follow-up

## Source and scope
FloW-Img, Cheng et al., ICCV 2021, "FloW: A Dataset and Benchmark for Floating Waste Detection in Inland Waters". https://openaccess.thecvf.com/content/ICCV2021/papers/Cheng_FloW_A_Dataset_and_Benchmark_for_Floating_Waste_Detection_in_ICCV_2021_paper.pdf

The user downloaded FloW_IMG.zip, now preserved in revision/flow/raw. ZIP CRC checks passed. Its SHA-256 is recorded in audit.json. It contains 2,000 JPEG images and 2,000 VOC XML annotations, with 1,200 images in training and 800 in test, consistent with the published 60/40 split. The downloaded annotations contain 5,272 bottle boxes; the publication reports 5,271. We preserve the supplied annotations and document this one-box discrepancy. No boxes required clipping or removal. XML xyxy bounds are interpreted directly as continuous coordinates, without an added/subtracted pixel. Image dimensions were verified. Hardlinked images must not be edited in place.

## Partition
All 800 original test images remain in test. Validation is derived exclusively from the official training pool using seed 20261028. Identical-image hashes are grouped before allocation; no byte-identical image crosses any partition. The final counts are:

| Split | Images | Bottle boxes |
|---|---:|---:|
| Training | 960 | 2,638 |
| Validation | 240 | 610 |
| Test | 800 | 2,024 |

The original split shares XML source-folder labels across training and test. These labels are not reliable independent sequence/site identifiers, and the partition must not be called site-independent or sequence-disjoint. Validation is a seeded random inner split, not a grouped-video holdout. The `frame` manifest field is an individual image ID, not a video ID. Exact duplicates were checked; this does not exclude near-duplicates or temporal dependence. This comparison follows the supplied benchmark split and cannot establish generalization to independent sites.

## Training
Four architectures: YOLO11n baseline, P2, P2–ADown, and P2–ADown–TFE without TinySCA. The code identifier p2_adown_tfe_off means attention off, NOT TFE off. Five seeds (0–4), all from scratch: 20 runs. All use the same partition and one native class, bottle; this does not reproduce the original two-class task or test pretrained initialization on FloW.

150 epochs, no early stopping, physical batch 4, 640-pixel input, workers=4, FP32 (AMP disabled), nominal batch 16 and deterministic settings. AdamW and augmentations are inherited explicitly from the original experiment.json and frozen in frozen_plan.json. Best weights are selected by validation fitness only. Separate two-epoch TFE pilots use the same 150-epoch schedule and intentionally stop after checkpoint 2 for timing; pilot checkpoints are not used in final results. No budget shortening is justified by these timing pilots.

The initial AMP pilot had an infinite validation classification loss in epoch 1 (finite in epoch 2). The FP32 pilot had finite training and validation losses in both epochs, with elapsed epoch times of 26.0009 and 22.3276 seconds. FP32 is therefore fixed for all 20 runs before final training, based on numerical stability, not test performance. Cache and warmup differ, so the pilots do not establish a causal speed advantage of FP32. At the second-epoch TFE rate, 20 runs of 150 epochs would take approximately 18.61 hours before test evaluation and bootstrap. Later timings may differ. The original AMP plan is archived with the pilot, not mixed with final runs.

## Evaluation
All final training precedes test inference. Report AP50, AP50–95, precision and recall, five-seed means and sample SD. One-class macro AP equals bottle AP. Native image dimensions are restored before bbox evaluation. Confidence floor 0.001, NMS IoU 0.70, maximum 300 detections. Fixed precision/recall uses confidence 0.25 and IoU 0.50. Secondary scale results use native pixel coordinates and are not directly comparable to original 640-pixel strata.

Three planned AP50–95 contrasts: P2 minus baseline, P2–ADown minus P2, TFE variant minus P2–ADown. Paired image bootstrap uses 1,000 identical draws across all models and seeds, averaging the five fitted seeds within each draw. Report pointwise 95% and Bonferroni 98.333% intervals for the three contrasts. These intervals condition on fitted models and assume independent images; related scenes may make them optimistic. Report seed SD separately and do not interpret non-significance as equivalence. Do not describe this as a source-video bootstrap.

## Execution
Use iniciar_flow_20.cmd for training, evaluation and analysis, or medir_flow.cmd for the separate timing pilot only. Interrupted training resumes from a valid last.pt; completed jobs are skipped. Errors do not silently change batch or configuration. Outputs are in V2/flow_supervised_640. RiSID and Qiao data remain preserved; their training commands must not be launched for this chosen FloW experiment.

The manuscript and reviewer letter are not yet changed to claim results. Integrate this protocol and numerical findings after completion and verification of all 20 runs.
