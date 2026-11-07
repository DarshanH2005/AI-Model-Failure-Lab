# AI Model Failure Lab — Shortcut Detective

A reproducible vision-model laboratory: train, intervene on nuisance cues, inspect failures, select repair data and compare repairs under equal labelled-image budgets.

The specification is [PROJECT.md](PROJECT.md). The live delivery status is [docs/status.md](docs/status.md). Hardware measurements are in [docs/hardware.md](docs/hardware.md). Only saved, completed experiments support performance claims.

## Setup and launch

Python 3.11+ and the dependencies in pyproject.toml are required. For CUDA on Windows, install a matching PyTorch build from the official PyTorch index before installing the project. The exact environment used here is pinned in requirements-gpu-lock.txt.

On this machine, a fresh CUDA environment is already available at LocalAppData/ShortcutDetective/venv-gpu. The launcher selects it automatically:

```powershell
./Start-Lab.ps1 -Mode test
./Start-Lab.ps1 -Mode smoke
./Start-Lab.ps1 -Mode dashboard
```

The machine has an RTX 4050 Laptop GPU with 6 GB VRAM, an i7-12650H and approximately 48 GB RAM. The original global Python installation contained CPU-only PyTorch; the isolated environment has working CUDA. Both required models fit the planned batch size 32 with AMP. GPU rental was not needed to fit the models.

For a different machine:

```powershell
python -m venv .venv
./.venv/Scripts/python -m pip install -r requirements-gpu-lock.txt
./.venv/Scripts/python -m pip install -e ".[dev]"
./.venv/Scripts/python -m pytest -q
./.venv/Scripts/python -m streamlit run app/streamlit_app.py
```

The lock targets Windows/Python 3.14/CUDA 13.0. On other platforms resolve compatible dependencies from pyproject.toml and record a new environment. Do not silently change dependencies for a frozen experiment.

Windows blocks Python writes directly inside this Documents folder. The launcher writes run artifacts under LocalAppData and copies them back through PowerShell. It leaves source files in this repository.

## Reproduction

```powershell
python -m failure_lab.cli generate --config configs/smoke.yaml
python -m failure_lab.cli validate-data --config configs/smoke.yaml
python -m failure_lab.cli run-experiments --config configs/smoke.yaml
python -m failure_lab.cli report --config configs/smoke.yaml
```

Use launcher mode pilot for the development pretrained ResNet pilot and core for the full matrix. The core protocol retains the planned split sizes, models, seeds and budgets. Run directories freeze source/configuration hashes. Completed individual conditions can resume from existing checkpoints; corrupt checkpoints and changed protocols fail. An interrupted optimisation condition restarts from its original checkpoint.

The comparable validation search and decorrelated controls are reproducible using:

```powershell
python -m failure_lab.pilot --config configs/pilot.yaml --root artifacts/tuning
```

All tuning/control code reads only training and validation. Core tests use a fresh registry namespace after pilot evaluation.

## Method and saved artifacts

A positive tile contains a scratch, crack or chipped edge. Background, corner symbol and brightness correlate with labels during biased training. Canonical edits preserve the true task; visibility checks reject invalid edits.

The two pretrained models are timm ResNet-18 and DeiT-Tiny. A small CNN is used only for CPU smoke tests. Training uses head warm-up, AdamW, cosine decay and challenge-validation worst-group checkpoint selection, with matched accuracy as tie-breaker.

Every selected candidate contributes four new images. Targeted and random nuisance repair share exactly the same edit policy. Standard random repair uses mirror, five-degree rotation and bounded contrast. Repair sampling is 1:1 original/new. Continuation and all repairs have the same optimisation-step budget and reset to the same original checkpoint.

Outputs include configuration/source hashes, dependency/hardware metadata, pretrained revisions/weight hashes, checkpoints, selected IDs, expanded repair manifests, audit correctness/deltas/quality, per-image test predictions, raw/calibrated metrics, 16-group counts, bootstrap intervals, per-seed paired comparisons and aggregate reports.

The primary metric is decorrelated-test worst-group accuracy. Report matched and other shifts, calibration and seed variability too. A negative result is acceptable.

## Dashboard and media

The dashboard reads saved artifacts, caches checkpoints, and explores only development examples. It shows original/edited images, mask checks, original/repaired probabilities, budget curves, group counts and downloadable reports. Experiments with a different renderer are hidden to prevent inconsistent re-rendered examples.

The export command creates development galleries, a static budget plot and an optional 2-minute-40-second captioned demo video:

```powershell
python -m failure_lab.export --experiment artifacts/core-v4 --output artifacts/core-v4/media --video
```

Matplotlib, imageio and imageio-ffmpeg are required for export. No notebook state is needed. The exploration notebook only reads saved results.

## Verification and limitations

Eight tests check deterministic generation, split separation, valid intervention boundaries, label isolation, exact budgets, known metrics, checkpoint round-trip, cache isolation and a tiny end-to-end study. They have passed in the original CPU environment and a fresh CUDA environment. The package builds as a wheel. Streamlit AppTest checks the app against saved results.

This is a deliberately controlled synthetic proxy. It does not establish industrial reliability, complete internal reasoning, human annotation savings or an architecture-only causal effect. Semantic near-duplicates remain possible despite disjoint seeds and exact-hash checks. Bootstrap intervals capture object sampling, not all training variability. Report every seed, uncertainty and unresolved failures.






## Final measured study

All 66 core-v4 conditions have completed. The original models score 100% accuracy across the four synthetic test environments. Targeted and random nuisance repair tie on decorrelated worst-group accuracy at every budget; the task provides no headroom for improvement. One random-standard DeiT repair loses one inverted-test prediction out of 1,024. See docs/results.md for full tables, calibration and uncertainty.

Delivered evidence lives in artifacts/core-v4: frozen protocol/environment, checkpoints, predictions, repair manifests, dataset images/masks, verification receipts and media. Optional natural-image validation and perceptual near-duplicate analysis remain future work.

## External Waterbirds validation

Run `./scripts/Run-Waterbirds.ps1` for the separate six-condition photographic-composite bird benchmark. The wrapper downloads the official archive into LocalAppData, extracts it with Python's safe data filter, verifies split integrity and runs the frozen protocol in docs/waterbirds_protocol.md. It packages checkpoints, predictions and report into artifacts/waterbirds-v1 and adds a Waterbirds dashboard page. Images/archive stay in LocalAppData for local research and education; they are excluded from the source bundle. No paid compute is used.

This is group-robustness validation on a separate bird task. Verified foreground masks are unavailable, so it does not claim intervention-repair transfer. Use official test groups, separate selection/calibration validation and three seeds per architecture. Consult docs/waterbirds_results.md for measured findings after the study completes.
