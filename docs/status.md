# Delivery status

The planned 66-condition synthetic core and the additional six-condition Waterbirds external study are complete: 72 completed conditions total. Both model architectures use three seeds. All checkpoints, saved predictions, group counts and measured scores passed artifact verification. Both dashboard pages load and result navigation works.

The core dataset contains 18,096 distinct image hashes and 54,288 valid canonical intervention checks. Its original models already attain 100% synthetic accuracy; targeted repair ties random nuisance repair on the primary metric. One random-standard repair loses one inverted-test prediction. The core report, galleries and 160-second captioned video are in artifacts/core-v4.

Waterbirds uses 11,788 photographic composites with official splits, separate checkpoint-selection/calibration validation and no exact image hashes crossing splits. Mean official test worst-group accuracy is 35.88% for ResNet-18 and 49.07% for DeiT-Tiny. Per-seed outcomes, calibration, adjusted/unweighted accuracy, uncertainty and observed errors are in docs/waterbirds_results.md and artifacts/waterbirds-v1. Test scores were not used to change training.

Eleven integrity/end-to-end tests pass. Core and external dashboard QA passed; plots and galleries were inspected visually. The full studies ran locally in the isolated CUDA environment on the RTX 4050; no rental was used.

Remaining limitations: perceptual near-duplicate analysis, validation on real industrial defect photos and intervention-repair transfer with verified foreground masks. Waterbirds contains natural photographs with deliberately composited backgrounds and is not factory deployment validation. Downloaded images stay local for research/education and are excluded from source bundles.
