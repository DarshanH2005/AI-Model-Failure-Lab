# Dataset card

Binary synthetic tile inspection. Positive tiles contain a scratch, irregular crack or dark chipped-edge area. Negative tiles contain faint benign marks. Foreground texture, defect location, rounded geometry, position and ±12-degree rotation are deterministic functions of each unique object seed.

Background colour, corner symbol colour and brightness correlate independently with class at 90% during biased training. Inverted evaluation uses 10%. Challenge, decorrelated and unfamiliar environments cover all 16 joint label/cue groups. Unfamiliar images use sinusoidal background textures, ellipse watermarks and bounded 0.84/1.16 brightness instead of 0.88/1.12.

IDs include registry namespace, split and object index. Final registry_seed=100000000 introduces fresh objects after pilot evaluation. Within each registry every object seed is disjoint across splits. Related renderings stay in their parent's split.

Background and watermark edits preserve foreground pixels exactly. Validity requires at least eight defect pixels, contrast exceeding 70 intensity units and foreground clipping below 1%. Standard augmentations also undergo visibility checks. Exact duplicate image hashes are checked during image export. Stratified development galleries support visual QA.

Manifest includes object/render seed, defect type, masks, cue attributes/families, generator version and hashes. Repair manifests record every additional image, selected parent and intervention policy. Only pixels enter classifiers.

This benchmark is synthetic and deliberately simple. Perceptually similar normal objects may occur; exact seed separation and noise do not guarantee semantic diversity. Perceptual nearest-neighbour analysis remains a limitation.


## External-study data

Waterbirds uses the official 11,788 photographic composites: 4,795 train, 1,199 validation and 5,794 test images. Official validation is deterministically divided into 598 checkpoint-selection and 601 calibration samples, stratified by label/background group. All images open successfully, paths are unique, and no exact image hash crosses an official split. Four groups represent landbird/waterbird crossed with land/water background.

Downloaded images remain in LocalAppData/ShortcutDetective/waterbirds-v1 for local non-commercial research/education. The archive and images are excluded from source bundles. This constructed benchmark is distinct from unmodified natural deployment data, and no verified foreground masks are supplied by this release. See waterbirds_protocol.md for sources and known annotation limitations.
