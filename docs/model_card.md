# Model card

Supported production research architectures: timm ResNet-18 a1 ImageNet weights and DeiT-Tiny fb ImageNet weights. A small randomly initialised CNN exists exclusively for CPU smoke verification.

Models predict defect probability using binary cross-entropy. Training includes head-only warm-up, AdamW and a cosine learning-rate schedule. Checkpoint selection uses challenge-validation worst-group accuracy, then matched-validation accuracy. Scalar temperature is fitted on separate calibration validation.

Models accept synthetic tiles at configured resolution with checkpoint-specific normalisation. They are not validated for industrial inspection. Pretraining recipes and model sizes differ; comparisons cannot isolate architecture effects. Calibration under distribution shift is measured rather than guaranteed.

Checkpoints, preprocessing, temperatures, hashes, histories and environments are saved in each run. Never load checkpoints from an untrusted source.


## Measured release

The core-v4 release contains 66 conditions across two architectures, three seeds and repair-image budgets 128/256/512. Original models achieve perfect accuracy on all four declared synthetic test environments. Targeted repair ties random nuisance repair on decorrelated worst-group accuracy. This is a ceiling-limited finding; it does not establish that either method will work on harder or real images. Consult `results.md` for calibration, uncertainty and the complete comparison.

## External model task

Waterbirds models use new bird-classification heads and three seeds per architecture under the separate frozen protocol. Their positive probability means waterbird, not surface defect. The checkpoints explicitly store this task, preprocessing, calibration temperature and training history; core defect checkpoints are kept separate. No foreground interventions or repair-transfer claim applies to this external study.
