# AI Model Failure Lab

> Discover what a vision model wrongly relies on, reproduce its failures, and evaluate whether targeted training data repairs them.

**Status:** Core release complete: 66 experiment conditions, verified artifacts, dashboard, report and captioned demo. See docs/results.md for the ceiling-limited result and one observed regression.  
**Target role:** AI/ML engineer.  
**Project type:** Applied deep learning research with an interactive demo.  
**Planning assumption:** One developer, about eight part-time weeks, with local or cloud GPU access. Timeline and hardware are not yet confirmed.  
**Demo name:** Shortcut Detective.

## 1. Objective

Build a reproducible laboratory that trains visual classifiers, discovers sensitivity to irrelevant image cues, and compares repair strategies under a fixed additional-data budget.

The first task is binary defect detection on synthetic product tiles. A visible surface defect determines the true label. Background, corner watermark, and mild lighting are deliberately correlated with that label during training. Evaluation breaks those correlations.

The complete workflow is:

**Train → audit → reproduce → select repair data → retrain → evaluate.**

The central research question:

> Does selecting examples by their sensitivity to controlled nuisance interventions improve robustness more than random selection, given the same additional training-image budget?

A valid negative result is acceptable. Completion means answering the question with reproducible evidence, explaining limitations, and presenting a working demonstration.

## 2. Value and originality

This project demonstrates dataset construction, supervised training, CNNs, vision transformers, experimental design, calibration, robustness, reproducibility, and deployment.

Its proposed contribution is an integrated failure-discovery and budgeted-repair workflow, supported by a controlled benchmark and fair comparisons. It does not claim to invent shortcut learning or model repair.

Shortcut learning describes models relying on predictive rules that work in their training environment but fail under changed conditions. Research foundation: [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780).

The eventual interview story:

> “I constructed a controlled benchmark, identified misleading cues, compared repair strategies under equal data budgets, and measured generalisation to conditions excluded from repair.”

## 3. Users and workflows

### ML practitioner

Select a supported checkpoint, audit its sensitivity, inspect paired examples, choose a repair budget, and compare measured results across strategies.

### Recruiter or interviewer

Change an irrelevant visual cue, observe the prediction, compare original and repaired models, then inspect aggregate results and unresolved failures.

### Developer

Reproduce the generator, training, repair, evaluation, and report from versioned configuration files.

## 4. Scope

### Required first release

- One synthetic task: defective versus non-defective product tile.
- Three nuisance factors: background, watermark, bounded lighting.
- Two architectures: ResNet-18 and DeiT-Tiny.
- Paired interventions with label-preservation checks.
- Per-factor sensitivity summaries and failure examples.
- Four repair conditions and three additional-data budgets.
- Three experiment seeds.
- Matched, decorrelated, inverted, and unfamiliar-condition tests.
- A dashboard using real saved results.
- Reproducible repository, experiment report, and demo video.

### Optional extensions

- Waterbirds external validation.
- Group DRO training baseline.
- Uncertainty-based example selection.
- A custom small CNN trained from scratch.
- Supplementary saliency visualisations.
- Hosted inference or dashboard.

### Outside the first release

Arbitrary uploaded models or images, general-purpose failure detection, GAN/diffusion editing, video, language-model auditing, reinforcement learning, and automatic production deployment.

Architectures should serve the experiment. RNNs and GANs do not need artificial features merely to appear in the technology list.

## 5. Dataset

### 5.1 True task

Render a product tile in a central foreground region. Positive examples contain a scratch, crack, or chipped area. Negative examples include normal texture and benign marks so the classifier cannot simply detect any line.

Store foreground and defect masks. Randomise geometry, texture, orientation, and bounded object position. Start with 224 × 224 RGB images rendered directly at input resolution.

This is a controlled visual-inspection proxy, not a realistic industrial benchmark.

### 5.2 Nuisance factors

| Factor | Intervention | Label-preservation requirement |
|---|---|---|
| Background | Replace colour/texture outside foreground | Product and defect pixels remain unchanged |
| Watermark | Add/remove/switch corner symbol | Symbol stays outside product mask |
| Lighting | Bounded illumination edit | Defect remains visible; excessive clipping is rejected |

Implement background and watermark first. Add lighting after visual inspection and automated visibility checks.

A pair preserves object geometry and defect identity. Save all intervention parameters. Define minimum defect size, contrast, and foreground-clipping bounds before experiments. Exclude ambiguous edits instead of counting them as shortcut failures.

### 5.3 Correlations and environments

Use balanced class labels. For each binary nuisance attribute `a`, start with `P(a = y) = 0.90` in training. Sample factors independently conditional on label and record realised distributions.

Include minority cue combinations rather than making any cue a deterministic label oracle.

- **Matched:** correlations resemble training.
- **Decorrelated:** nuisance attributes are independent of label, with balanced joint groups.
- **Inverted:** nuisance-label associations reverse.
- **Unfamiliar:** new texture families, watermark designs, or bounded lighting values; maintain explicit binary attribute mappings.

Evaluate single-factor changes separately from joint changes. Joint failure alone cannot identify which factor mattered.

### 5.4 Initial split sizes

These are starting values to revise during a small pilot, before final tests are frozen.

| Split | Unique base objects | Purpose |
|---|---:|---|
| Training | 6,000 | Initial supervised training |
| Matched validation | 1,000 | Ordinary learning checks |
| Challenge validation | 1,000 | Development and checkpoint selection |
| Calibration validation | 1,000 | Temperature fitting |
| Development audit | 1,000 | Failure discovery and demo examples |
| Repair candidate pool | 4,000 | New training-example selection |
| Matched test | 1,024 | Final ordinary performance |
| Decorrelated test | 1,024 | Balanced-group robustness |
| Inverted test | 1,024 | Correlation reversal |
| Unfamiliar test | 1,024 | Transfer beyond development conditions |

Paired renderings are related observations, not additional independent objects.

### 5.5 Leakage prevention

1. Assign base objects to splits before rendering variants.
2. Keep every rendering of one object in that split.
3. Use disjoint object seeds and selected held-out texture families.
4. Check object identifiers, exact hashes, and visually similar samples across splits.
5. Keep final test images and labels out of tuning, auditing, and selection.
6. Freeze the final protocol before running final tests.
7. If final-test findings motivate a change, preserve those results and introduce a fresh holdout for subsequent claims.

### 5.6 Manifest

```text
image_id, base_object_id, split, label, render_seed,
defect_type, defect_parameters, foreground_mask_path, defect_mask_path,
background_attribute, background_family, watermark_attribute,
watermark_design, lighting_attribute, lighting_parameters,
parent_image_id, intervention_factor, intervention_parameters,
generator_version, image_path, image_hash
```

The evaluator may read labels and group metadata. Candidate selection receives images and allowed transformations, not ground-truth labels or a label-based oracle ranking.

## 6. Models and training

| Model | Role |
|---|---|
| ResNet-18 | Manageable CNN baseline |
| DeiT-Tiny | Manageable vision-transformer comparison |

Use ImageNet-pretrained checkpoints and replace the classifier with a binary output. Write the training and evaluation loops yourself. Record exact checkpoint revisions and model-specific preprocessing.

Implementation references: [timm ResNet documentation](https://huggingface.co/docs/timm/main/models/resnet), [DeiT-Tiny model card](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k).

These models differ in parameter count and pretraining recipe. Results compare these configurations; they do not isolate architecture as the sole cause.

### Training protocol

- Binary cross-entropy with logits; sigmoid probabilities for reporting.
- Brief classifier-head warm-up, followed by backbone fine-tuning.
- AdamW with a documented schedule.
- A small declared validation search over learning rate and weight decay.
- Comparable tuning budgets and identical splits across architectures.
- No generic colour/background augmentation in the initial biased-training condition.
- Save optimisation steps, batch size, normalisation, precision, duration, and memory use.
- Select checkpoints using challenge-validation worst-group accuracy, with matched accuracy as a tie-breaker.
- Do not select an artificially bad checkpoint to create a dramatic demo.

### Calibration

Fit scalar temperature on the separate calibration-validation split after checkpoint selection. Report both raw and calibrated probability metrics. Keep main selection scoring on raw probabilities so calibration does not silently alter the method.

Temperature scaling does not guarantee calibration under every shift. Reference: [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).

## 7. Failure investigation

### Paired intervention

For image `x`, generate label-preserving variant `T_f(x)` for factor `f`. Use the same frozen model for both.

Record defect probability, predicted class, class changes, correctness when evaluation labels are available, and transformation-quality results.

Controlled edits establish sensitivity to a factor in this environment. They do not establish the model's complete internal reasoning or real-world causality.

### Sensitivity score

For defect probability `p`:

```text
delta_f(x) = abs(p(T_f(x)) - p(x))
score(x) = mean over factors f of delta_f(x)
```

Use one fixed canonical intervention per factor initially. Retain factor-level deltas.

Report mean probability change, class-flip rate, and correctness changes separately. A flip is not necessarily an error; a stable prediction can still be incorrect.

### Audit report

Include factor rankings, confidence-change distributions, group performance, harmful/beneficial/neutral examples, quality checks, configuration, checkpoint identity, and a plain-language interpretation.

Optional saliency or attention views supplement the evidence. A heatmap alone is not proof of dependence.

## 8. Budgeted repair

### 8.1 Budget definition

Use `B ∈ {128, 256, 512}` additional labelled training images. For each selected parent, create `k = 4` renderings: one original and three additional variants.

Each data-adding method selects `B / k` previously unused base objects and receives exactly `B` images. Do not count duplicate parents twice.

The simulator supplies labels automatically. This is a training-data-budget experiment, not evidence of savings in human annotation cost. Report candidate-scoring compute separately.

### 8.2 Conditions

| Condition | Parent selection | Extra data | Purpose |
|---|---|---|---|
| Continuation control | None | Reuse original training data | Control for further optimisation |
| Random + standard augmentation | Uniform random | Original plus three conventional variants | Practical baseline |
| Random + nuisance intervention | Uniform random | Original plus one edit per nuisance | Control for intervention policy |
| Targeted + nuisance intervention | Highest sensitivity | Same nuisance edit policy | Test targeted selection |

The primary comparison is targeted versus random **with the same nuisance intervention policy**. Comparing targeted repair against standard augmentation changes two things and is therefore secondary.

Choose standard transforms in advance and check defect visibility. Use independent random selections across experiment seeds.

### 8.3 Procedure

1. Start each condition from the same original checkpoint for its seed.
2. Score candidates with the original frozen model.
3. Select parents without reading labels.
4. Request generator labels after selection.
5. Render and validate repair examples.
6. Mix original and repair examples during fine-tuning.
7. Fix optimisation-step count, optimiser policy, and sampling ratio across data-adding conditions.
8. Give the continuation control the same step count using original data.
9. Choose repair settings using validation, then freeze them.
10. Evaluate final checkpoints on identical protected tests.

Start with a 1:1 original-to-repair sampling ratio. More data increases diversity, not optimisation steps. This separates data budget from longer training.

### 8.4 Experiment matrix

- Two architectures.
- Three data-adding methods.
- Three budgets.
- Three complete experiment seeds.

Total: 54 repair runs, six initial training runs, and six continuation controls. Reuse each continuation control across budgets because it adds no data.

First run a pilot with one architecture, one budget, and one seed. Expand only when data, metrics, and budget accounting are verified.

### 8.5 Later ablations

Prioritise one-factor repair, mean versus maximum sensitivity, and training correlation strength. Do not require a full Cartesian product of all extensions.

Optional Group DRO assumes known groups; report that assumption explicitly. Reference: [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731).

## 9. Evaluation

### Primary metric

Worst-group accuracy on the decorrelated held-out test, using label and three binary nuisance attributes: 16 joint groups. Report sample counts for every group.

For unfamiliar conditions, retain explicit group mappings and additionally report new texture/design families.

### Supporting metrics

| Metric | Meaning |
|---|---|
| Matched accuracy | Ordinary performance |
| Balanced accuracy and defect recall | Class balance and missed defects |
| Inverted/unfamiliar accuracy | Robustness across declared shifts |
| Worst-group accuracy | Weakest predefined subgroup |
| Class-flip rate | Instability under valid edits |
| Mean probability change | Sensitivity magnitude |
| Negative log-likelihood and Brier score | Probability quality |
| Expected calibration error | Binned confidence/accuracy agreement |
| Gain versus additional-data budget | Repair efficiency |
| Latency and peak memory | Runtime practicality |

Fix the decision threshold using validation; initially use 0.5. Document calibration bins and counts, and avoid relying only on ECE.

### Statistical reporting

- Report every seed and mean ± standard deviation.
- Compare paired method differences on identical seeds and examples.
- Bootstrap by base object for related image variants; recompute the full statistic, including minimum group accuracy.
- State that per-checkpoint bootstrap intervals do not capture all training variability.
- Include inconclusive results and sample sizes.
- Never present placeholder improvements as findings.

### Completion criteria

The project succeeds when generation is reproducible, failures or their absence are explained, the required comparisons are complete, held-out trade-offs are reported, and the working demo agrees with saved artifacts.

A positive gain over random nuisance-intervention selection is an aspiration, not a promised acceptance threshold. A rigorous negative result can complete the project.

## 10. Architecture

```text
Generator + configuration
         |
         v
Versioned manifests + split registry
         |
         +--> Trainer --> Original checkpoints
         |                    |
         +--> Development auditor
         |                    |
         |                    v
         |           Sensitivity + selector
         |                    |
         +--> Repair renderer-+
                   |
                   v
             Repair trainer --> Repaired checkpoints
                                      |
Protected test registry --------------+
                   |
                   v
           Evaluation artifacts
                   |
                   v
           Report + dashboard
```

| Component | Input | Output |
|---|---|---|
| Generator | Seed, object, environment | Images, masks, manifest |
| Loader | Manifest and preprocessing | Image tensors |
| Trainer | Configuration and training split | Checkpoint, history, metadata |
| Interventions | Object and factor settings | Valid paired rendering |
| Auditor | Checkpoint and development images | Sensitivity records |
| Selector | Unlabelled candidates | Parent IDs and scores |
| Repair trainer | Original checkpoint and new images | Repaired checkpoint |
| Evaluator | Checkpoint and frozen registry | Predictions and metrics |
| Dashboard | Saved artifacts and supported models | Interactive investigation |

Dashboard metrics must come from machine-generated artifacts with run identifiers.

## 11. Technology

| Area | Choice |
|---|---|
| Language | Python |
| Models | PyTorch and timm |
| Rendering | Pillow and NumPy |
| Data/metrics | pandas and scikit-learn |
| Charts | Plotly and Matplotlib |
| Demo | Streamlit |
| Configuration | YAML with validated Python objects |
| Tracking | Local JSON/CSV initially; optional MLflow |
| Verification | pytest |
| Packaging | pyproject.toml and dependency lock |

Resolve compatible versions during implementation and record Python, PyTorch, CUDA, hardware, and checkpoint revisions. A separate frontend, database, queue, or LLM is unnecessary initially.

## 12. Repository structure

```text
ai-model-failure-lab/
  PROJECT.md
  README.md
  pyproject.toml
  configs/
    dataset.yaml
    models/
    audit.yaml
    repair.yaml
    experiments.yaml
  src/failure_lab/
    data/
      generator.py
      manifests.py
      splits.py
      dataset.py
    models/factory.py
    training/
      train.py
      repair.py
      checkpoints.py
    auditing/
      interventions.py
      sensitivity.py
      reports.py
    selection/
      random.py
      targeted.py
    evaluation/
      metrics.py
      calibration.py
      evaluate.py
    cli.py
  app/streamlit_app.py
  tests/
  notebooks/exploratory_analysis.ipynb
  data/
    manifests/
    generated/
  artifacts/
    checkpoints/
    predictions/
    audits/
    reports/
  docs/
    dataset_card.md
    model_card.md
    experiment_protocol.md
    results.md
    demo_script.md
```

Notebooks are for exploration; core workflows live in importable modules. Keep large generated images and checkpoints out of ordinary Git history.

## 13. Proposed CLI

These commands describe software to implement; they do not exist yet.

```bash
python -m failure_lab.cli generate --config configs/dataset.yaml
python -m failure_lab.cli validate-data --config configs/dataset.yaml
python -m failure_lab.cli train --config configs/models/resnet18.yaml --seed 11
python -m failure_lab.cli audit --run-id resnet18-seed11 --split audit_dev
python -m failure_lab.cli repair --run-id resnet18-seed11 --method targeted --budget 256
python -m failure_lab.cli run-experiments --config configs/experiments.yaml
python -m failure_lab.cli evaluate --experiment-id core-v1 --registry frozen-test-v1
python -m failure_lab.cli report --experiment-id core-v1
streamlit run app/streamlit_app.py
```

Each command writes configuration hash, source revision, seed, dependency versions, checkpoint identity, inputs, and timestamps. Invalid split use or incompatible preprocessing should fail clearly.

## 14. Dashboard

### Overview

Explain the true task, environment, model, run status, and whether displayed results are development or final evaluation.

### Investigation

Show original and edited images, valid nuisance controls, predictions, defect probability, probability delta, true label, and transformation checks. Offer foreground/defect-mask views.

### Repair comparison

Show original/repaired predictions, budget curves, group performance, seed variability, ordinary-accuracy trade-offs, and galleries of successes and unresolved failures.

### Report

Expose run ID, configuration, evaluation environment, and downloadable summaries. Deterministic explanations suffice; no LLM narrator is required.

Run training from the CLI. Cache dashboard models and load saved results so the recruiter does not wait for retraining. Use development examples for repeated interactive exploration.

## 15. Verification

Test properties that can invalidate the study:

- Deterministic generation from a seed/configuration.
- No cross-split base-object overlap.
- Intervention boundaries and visibility checks.
- Exact budget counts and unique parent selection.
- Selection cannot access labels or final test artifacts.
- Known-value metric checks.
- Checkpoint reload and preprocessing consistency.
- Tiny end-to-end generation → train → audit → repair → evaluate run.

Inspect stratified image galleries before full training. Verify filenames and manifest metadata do not enter the classifier as label clues.

Final QA includes clean-environment reproduction and demo operation without hidden notebook state.

## 16. Milestones

| Milestone | Indicative window | Deliverable | Gate |
|---|---|---|---|
| Generator and protocol | Week 1 | Renderer, masks, manifests, splits | Visible defects, valid pairs, no overlap |
| CNN baseline | Week 2 | Training and evaluation | Task learnable; smoke pipeline passes |
| Investigation | Week 3 | Interventions and report | Failures reproducible or setup assessed honestly |
| Repair pilot | Week 4 | Four conditions at one budget | Equal data counts and matched steps |
| Transformer | Week 5 | Second model and pilot | Comparable documented protocol |
| Main experiments | Week 6 | All budgets and seeds | Frozen settings, traceable outputs |
| Analysis and demo | Week 7 | Dashboard and report | Displays agree with artifacts |
| Portfolio release | Week 8 | Repository and video | Clean reproduction and checklist pass |

These windows are estimates. Reduce optional work first if time or compute is constrained. Describe a reduced study accurately rather than calling a one-seed pilot a completed multi-seed experiment.

## 17. Compute strategy

Start with a small CPU-friendly smoke test and roughly 1,000 training objects for the pilot. Measure one training run, candidate audit, and repair run before estimating the full matrix.

Use GPU fine-tuning, measured batch sizes, and mixed precision where supported. Record the precision policy. Generate images lazily if disk usage is high.

Frozen-backbone results are suitable for a prototype but must be labelled separately. Cost and runtime remain unestimated until measured; CPU-only access may require a smaller study.

## 18. Risks

| Risk | Response |
|---|---|
| Synthetic task too easy | Add benign marks and geometry variation; disclose limits |
| Shortcuts fail to emerge | Inspect cue visibility and correlations using development data |
| Edits hide defects | Reject invalid variants and inspect examples |
| Selected examples redundant | Limit parents; inspect diversity; later test diversity-aware selection |
| Targeted method loses | Report and investigate alignment between sensitivity and errors |
| Architecture comparison confounded | Record pretraining, size, and tuning differences |
| Repair harms another group | Report all groups and matched-performance trade-offs |
| Test informs tuning | Enforce protected access and preserve experiment history |
| Experiment cost grows | Pilot first; defer optional ablations |

## 19. External validation

Waterbirds provides a separate bird-classification task with background correlations. Reference: [WILDS dataset documentation](https://wilds.stanford.edu/datasets/).

Check access and redistribution conditions, respect official splits, and train a separate task head. Use paired edits only when masks and label-preservation checks are available. Report natural-image results separately.

If valid edits are unavailable, evaluate groups without claiming the intervention-and-repair method transferred. Transfer failure is a substantive result.

## 20. Portfolio deliverables

- Repository with setup, quickstart, architecture, and reproduction instructions.
- Dataset card explaining generation, correlations, splits, and limitations.
- Model card documenting checkpoints, intended use, and failures.
- Protocol written before final evaluation.
- Results report with tables, budget curves, uncertainty, and failure galleries.
- Demo checkpoints and a lightweight launch path.
- Two-to-three-minute demo video.
- Resume entry using measured results only.

### Demo sequence

1. Explain the defect signal and misleading training cues.
2. Change a nuisance factor on a development example.
3. Compare original and repaired predictions.
4. Show aggregate held-out results against random selection.
5. Show a failure or regression and the experiment it suggests next.

### Resume template

> Built an interactive vision-model auditing lab with PyTorch, ResNet-18, and DeiT-Tiny; designed controlled nuisance interventions and compared targeted versus random repair across [N] reproducible runs, achieving [measured result] on [named held-out metric] at an additional-data budget of [B].

If targeted repair does not win, replace the achievement with the meaningful finding. Do not imply that pretrained architectures or existing methods were invented here.

## 21. Completion checklist

- [x] Scope and protocol frozen before final evaluation.
- [x] Masks, labels, visibility, and split boundaries verified.
- [x] Original models and all required conditions complete.
- [x] Data counts and optimisation comparisons valid.
- [x] Both architectures, three budgets, and three seeds reported.
- [x] Final tests excluded from tuning and selection.
- [x] Group counts, variability, calibration, and trade-offs included.
- [x] Dashboard matches saved results.
- [x] Report and demo include limitations and failures.
- [x] Clean source-copy integrity/end-to-end reproduction passes (8 tests); saved release reloads both models in the dashboard. The full 66-run matrix was executed once in the isolated CUDA environment.
- [x] README, cards, report, and video complete.
- [x] Resume claims are measured and scoped accurately.

## 22. First implementation steps

1. Create the package, configuration files, and manifest schema.
2. Implement tiles, defects, and masks before misleading correlations.
3. Inspect a small balanced sample and valid intervention pairs.
4. Add split-integrity and rendering checks.
5. Train a CNN in a decorrelated control environment to verify the real task is learnable.
6. Train the biased environment and audit development conditions.
7. Implement budgeted repair before the dashboard and optional methods.

## 23. References

- [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780).
- [Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731).
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
- [WILDS dataset documentation](https://wilds.stanford.edu/datasets/).
- [timm ResNet documentation](https://huggingface.co/docs/timm/main/models/resnet).
- [DeiT-Tiny model card](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k).

Dataset recipes, scoring rules, experiment counts, milestones, and software contracts above are proposed project decisions, not published performance claims.



## Completed external validation

The additional planned Waterbirds group-robustness check is complete: six independently trained conditions, official splits, three seeds per architecture and no test-driven tuning. See docs/waterbirds_results.md. Valid foreground masks were unavailable, so this release evaluates groups without claiming that the intervention-and-repair method transferred. Real industrial defect validation and perceptual near-duplicate analysis remain limitations.
