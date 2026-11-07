# Local hardware and measured compute

Detected 5 October 2026:
- NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB VRAM (nvidia-smi).
- NVIDIA driver 617.14, CUDA driver capability 13.4.
- Intel Core i7-12650H, 10 cores / 16 logical processors.
- 51,210,870,784 bytes installed RAM (approximately 47.7 GiB).
- More than 100 GiB free on the system drive at setup.

The global Python environment had CPU-only PyTorch. This did not indicate an absence of GPU hardware. A fresh isolated environment at LocalAppData/ShortcutDetective/venv-gpu now uses PyTorch 2.14.1+cu130 and torchvision 0.29.1+cu130. CUDA availability and a GPU matrix multiplication were verified.

Synthetic steady-state training benchmark, 224×224, batch 32, AdamW, AMP:
| Model | Seconds per step | Peak allocated tensor memory |
|---|---:|---:|
| ResNet-18 | 0.0447 | 618.8 MiB |
| DeiT-Tiny | 0.0342 | 691.8 MiB |

These figures exclude dataset rendering, transfers, auditing, validation, calibration and process/context overhead. They are feasibility measurements, not total-runtime promises. See hardware-benchmark.json for machine-readable values.

The planned matrix fits locally. Rental is optional for faster turnaround, not required by the measured memory needs. No rental or paid service was purchased.

