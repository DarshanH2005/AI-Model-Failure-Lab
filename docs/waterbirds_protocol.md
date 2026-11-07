# Waterbirds external validation protocol v1

Frozen before model training or test prediction. This is a separate photographic-composite bird task, not real factory defect validation or a transfer claim for synthetic intervention repair.

Source: https://github.com/kohpangwei/group_DRO and https://github.com/p-lambda/wilds/blob/main/wilds/datasets/waterbirds_dataset.py
Archive: https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz
Use: non-commercial research and educational purposes. Downloaded images remain locally stored; source bundle excludes them.

Use official metadata split 0=train, 1=validation, 2=test. Verify file existence, distinct paths and image hashes across splits. Deterministically divide official validation equally within each label/background group into selection and calibration using seed 20261005. No test examples or labels enter training, checkpoint selection, calibration or hyperparameter choice.

Architectures: resnet18.a1_in1k and deit_tiny_patch16_224.fb_in1k, pretrained ImageNet, new binary head. Seeds 11/22/33. All official training images. Resize RGB to 224x224 with bicubic interpolation (no random crop), model-specific normalisation. No foreground interventions because verified masks are unavailable. Train 1500 steps, head warmup 100, batch 32, AdamW LR 1e-4 and weight decay .01, cosine decay, CUDA AMP. These choices are fixed from the core setup without Waterbirds test tuning. Checkpoint selection at steps 500/1000/1500 by selection-validation four-group minimum accuracy, overall accuracy tie-breaker, earliest exact tie. Fit scalar temperature on the separate calibration partition.

Six conditions, no repair claims. Evaluate official test once per selected model. Report unweighted accuracy, balanced accuracy, four-group accuracy/counts, worst-group accuracy, training-group-weighted adjusted accuracy (distinct from unweighted accuracy), raw and calibrated NLL/Brier/ECE, all seed outcomes and mean/sample SD. Object bootstrap 1000 repetitions recomputes minimum groups and excludes draws missing groups; intervals describe sampling uncertainty only. Save checkpoint hashes, source/environment/protocol hashes, validation histories and individual predictions. Test results must not trigger tuning within v1.

Known limitations: Waterbirds is constructed using natural photographs, not an untouched natural deployment dataset. Official bird labels include known species annotation issues. Architectures differ in pretraining. This protocol is a finite training-budget external check, not a reproduced published state-of-the-art result. Six fresh models assess generalisation on this separate task, not defect-checkpoint transfer. A negative result is retained.
