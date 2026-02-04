# Multimodal Augmentations (Audio / Visual)

A small collection of data augmentation implementations designed to **increase input diversity** and improve robustness when training **multimodal models** (e.g., audio-visual pipelines).

- **Visual**: a *yaw / pitch / roll* based transform that produces a **pseudo 3D-like rotation** (orthographic transform + directional perspective heuristic)
- **Audio**: **FilterAugment** — applies a **frequency-dependent random gain curve** to log-mel / spectrogram features

> This repository currently ships as **Jupyter notebooks (`.ipynb`)**. For production training pipelines, copy the relevant classes/functions into `.py` modules and import them.

---

## Repository Layout

- `src/rotation_aug.ipynb`
  - `RandomFace3DRotationOrthoDirPersp` (PyTorch `nn.Module`)
  - `rotate_with_perspective`, `partial_perspective`, `make_2d_rot_matrix`, etc.

- `src/filter_aug.ipynb`
  - `FilterAugment` (PyTorch `nn.Module`)
  - Includes a demo `ToySineWaveDataset` + mel-spectrogram visualization code

---

## Requirements

The notebooks use:

- Python 3.11+
- `torch`, `torchvision`
- `torchaudio` (FilterAugment demo)
- `numpy`, `matplotlib`
- `opencv-python` (image loading in the rotation demo)

Install example:

```bash
pip install torch torchvision torchaudio numpy matplotlib opencv-python
```

---

## Quickstart

### 1) Run the notebooks

```bash
jupyter lab
# or
jupyter notebook
```

Open each notebook and run cells from top to bottom to reproduce the examples/visualizations.

### 2) Use inside your training pipeline

Recommended workflow:

1. Copy the augmentation code from the notebook into a `.py` file.
2. Import and compose it with your existing preprocessing / transforms.

- Visual: `RandomFace3DRotationOrthoDirPersp`, `rotate_with_perspective` (+ helper functions)
- Audio: `FilterAugment`

---

## Visual: Pseudo 3D Rotation (Orthographic + Directional Perspective)

The core idea in `rotation_aug.ipynb`:

### What it does

- **Orthographic component**
  - yaw: scales the *x* axis by `cos(yaw)`
  - pitch: scales the *y* axis by `cos(pitch)`
  - roll: standard 2D rotation
- **Directional perspective (heuristic)**
  - adds the impression that the “far side” shrinks more than the “near side”
  - controlled by `yaw_persp_strength` and `pitch_persp_strength`

Implementation uses `torch.nn.functional.grid_sample`.

### Supported input shapes

- single image: `(C, H, W)`
- batch: `(B, C, H, W)`

### Example usage

```python
import torch
from torchvision.transforms import Compose, Pad, CenterCrop

# Assume you moved code from the notebook into rotation_aug.py
from rotation_aug import RandomFace3DRotationOrthoDirPersp

x = torch.rand(3, 224, 224)  # (C,H,W) or (B,C,H,W)

aug = Compose([
    Pad(8, fill=0, padding_mode="constant"),
    RandomFace3DRotationOrthoDirPersp(
        yaw_range=(-20.0, 20.0),
        pitch_range=(-10.0, 10.0),
        roll_range=(-10.0, 10.0),
        p=0.8,
        padding_mode="zeros",   # grid_sample padding_mode
        degrees=True,
        yaw_persp_strength=0.3,
        pitch_persp_strength=0.3,
    ),
    CenterCrop((224, 224)),
])

x_aug = aug(x)
```

### Parameters

- `yaw_range`, `pitch_range`, `roll_range`: rotation ranges (degrees by default)
- `p`: probability of applying the transform (supports per-sample masking inside a batch)
- `padding_mode`: `grid_sample` padding (`"zeros"`, `"border"`, etc.)
- `yaw_persp_strength`, `pitch_persp_strength`: directional perspective strength (0 disables the effect)

### Tips

- **Pad → Augment → Crop** is recommended to avoid cutting off edges after warping.
- The notebook uses `grid_sample(align_corners=True)`. If your codebase uses a different convention, consider aligning it across all warps/resizes.
- If the input is an integer tensor (e.g., 0..255), the code internally casts to float, scales to 0..1, then casts back.

---

## Audio: FilterAugment (Frequency-Dependent Random Filter)

`FilterAugment` in `filter_aug.ipynb` creates a random gain curve along the **frequency axis (F)** and applies it to a spectrogram-like feature.

- input shape: `(..., n_freq, n_time)`
  - e.g., `(B, F, T)`, `(B, C, F, T)`, `(F, T)`
- applied only in **training mode** (`module.train()`) and with probability `p`
- the most recently sampled filter (in dB) is stored in `FilterAugment.filt_db` for debugging/visualization

### Example (apply to log-mel in dB)

```python
import torch
import torchaudio

# Assume you moved code from the notebook into filter_aug.py
from filter_aug import FilterAugment

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=320, n_mels=64
)
amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

aug = FilterAugment(
    db_range=(-20.0, 20.0),
    n_band_range=(2, 5),
    min_bandwidth=4,
    filter_type="linear",   # "step" | "linear"
    input_in_db=True,       # dB input -> additive
    p=0.8,
)
aug.train()

wave = torch.randn(8, 1, 16000)          # (B, 1, T)
spec_db = amp_to_db(mel(wave))           # (B, 1, F, T)
spec_db_aug = aug(spec_db)

last_filt_db = aug.filt_db               # (B_flat, F, 1)
```

### Parameters

- `db_range`: per-band gain sampling range (dB)
- `n_band_range`: number of frequency bands `(min, max)`
- `min_bandwidth`: minimum band width (in bins)
- `filter_type`
  - `"step"`: constant gain per band
  - `"linear"`: sample gains at boundaries and linearly interpolate within each band
- `input_in_db`
  - `True`: assumes dB scale → `x + filt_db`
  - `False`: assumes linear scale → `x * 10^(filt_db/20)` (amplitude convention)
- `p`: probability of applying the augmentation (only in training mode)

> Note: if your input is a **power** spectrogram, you may prefer using `/10` instead of `/20` when converting dB to linear gain. The current implementation uses the amplitude convention (`/20`).

---

## Reproducibility (Optional)

If you want deterministic sampling:

```python
import random
import torch

random.seed(0)
torch.manual_seed(0)
```

---

## Limitations / Known Issues

- The visual transform is a **heuristic** and not a physically accurate 3D projection model.
- `rotation_aug.ipynb` references a demo image path (e.g., `../assets/example.png`) that is **not included** in this repo; update it to match your local environment.
- In `RandomFace3DRotationOrthoDirPersp.forward`, angle sampling currently uses `random.sample(...)` on a `(min, max)` tuple, which only picks **endpoints** and then repeats them across the batch. For continuous uniform sampling, replace it with per-sample `torch.empty(B).uniform_(low, high)` (a helper exists but is unused).
- Minimal packaging/tests: the code is provided as notebooks for experimentation, so you’ll likely want to refactor into modules and add unit tests before long-term use.

---

## Notes

- This code is meant for research/experimentation. You’ll likely need to tune `*_range`, `*_strength`, and `db_range` for your data/task.
