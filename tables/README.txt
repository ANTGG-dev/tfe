Supplementary experimental results (S1-S16)

S1: AP25/AP50/AP50-95 pointwise percentile 95% intervals for all 30 main combinations.
S2-S3: Main means, sample SD and per-seed metrics.
S4: All paired bootstrap AP contrasts; shared resampling draws.
S5: All exploratory Welch/BH tests; valid families: 868 scratch, 224 pretrained.
S6-S7: Grouped CV per fold and summary (baseline, P2, P2-ADown).
S8-S9: Physical batch sensitivity (paired t intervals/tests and per-seed values).
S10-S12: RiSID external metrics, summary and annotation-boundary sensitivity.
S13: Fixed-threshold precision/recall CIs, all main and batch8 configurations.
S14: Completion and retained-console audit, 235 training runs and 20 external evaluations.
S15: Retained incident register; not an exhaustive operational history.
S16: Separate code/configuration archive S16_reproducibility_package.zip.

S1 and S13 use 1000 shared resamples, averaging five fitted seeds per draw.
Source-name family bootstrap is primary for local test (10 inferred families);
image bootstrap is supplementary. Families are not verified flights. These
pointwise intervals are conditional on fitted seeds, not simultaneous and not
full uncertainty over future training. S13 contains 1920 rows (30 main plus
2 batch8 configurations x 5 strata x 3 class summaries x 2 metrics x 2 units).
P/R uses confidence >=0.25 and IoU >=0.50, preserving COCO ignored GT/detections.
Counts are pooled within each draw before taking ratios. P=0 for no detections;
R is unavailable for zero GT. Valid draw counts are provided. 'macro' means
pooled P/R across classes; for AP it means mean across available classes.
Fractions in S1-S7/S13 should be multiplied by 100 for percentages.
See README_followup.txt for the units of S8-S12. RiSID SD is between training
seeds, not independent-video uncertainty. No onboard energy was measured.
