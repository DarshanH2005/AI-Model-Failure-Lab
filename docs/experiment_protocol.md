# Frozen core protocol

The original split sizes and required matrix are retained in configs/experiments.yaml. Both pretrained architectures use seeds 11/22/33; budgets 128/256/512; original, continuation and three adding methods. Total selected checkpoints: 66.

Primary comparison: targeted nuisance versus random nuisance. Primary metric: raw-probability worst-group accuracy across 16 label/background/watermark/lighting groups on decorrelated evaluation.

Each selected parent contributes original plus three renderings. B/4 unique candidate parents contribute exactly B new images. Random nuisance and targeted nuisance use identical edits. Random standard and random nuisance share random parents within each seed/budget for a tighter paired policy comparison. Candidate ranking accepts only IDs and raw sensitivity scores; labels are attached after selection.

All repair conditions reset to the original selected checkpoint. AdamW and cosine schedule restart under the same policy. Repair batches are 1:1 original/new; continuation uses only original data. Every repair/control uses 300 steps. Initial training uses 1,500 steps, including 100 head-only warm-up steps. Batch size is 32 with CUDA AMP.

Comparable development search: two learning rates (1e-4, 3e-4) × two weight decays (0.01, 0.1), four equal-step trials per architecture. Both select 1e-4 and 0.01 using challenge WGA then matched accuracy. Separate decorrelated training controls verified task learnability. Search/control artifacts are preserved.

Source hash and configuration freeze precede core training/evaluation. Final registry_seed=100000000 separates final objects from pilot evaluations. Final tests never enter selection or tuning. Challenge validation chooses checkpoints by WGA with matched accuracy tie-breaker; temperature is fitted on calibration validation. Threshold is 0.5.

Every final test contains one rendering per independent object. Bootstrap resamples objects and recomputes the full minimum-group statistic; resamples missing an observed group are discarded. Individual intervals exclude training variability. Paired differences use identical resampled objects; all seed outcomes and sample SDs are reported.

Controls establish sensitivity in this synthetic environment, not complete internal reasoning. Pretraining recipes differ between architectures. A negative or inconclusive result is acceptable. Claims must cite measured final artifacts.

