# Video2Unity
### Transformer-Based Motion Refinement

**From monocular video to refined 3D motion for Unity animation.**

An AI & ML course project at Chung-Ang University (Fall 2025), exploring how temporal pose refinement and geometric post-processing can improve video-driven character animation.

**Author:** Hyunkyu Kang · [Website](https://khk0606.github.io/)  
**Stack:** Python · MediaPipe / BlazePose · TensorFlow / Keras · NumPy / SciPy · Unity

> This repository contains the Python pipeline, saved model artifacts, extracted keypoints, and example CSV outputs. The Unity scene and C# animation scripts described in the course report are not included.

## Overview

Frame-wise pose estimates can produce visible jitter, unstable ground contact, and limited global movement when used directly for character animation. This project combines a **30-frame Transformer refinement model** with **root-centered normalization, smoothing, and floor alignment** to prepare motion for a Unity dance performance.

The goal is a practical video-to-animation workflow without a dedicated marker-based motion-capture setup. It is a course-project prototype, not a calibrated motion-capture system or a validated benchmark release.

## Pipeline

```text
Video
  │
  ▼
MediaPipe / BlazePose ── 33 landmarks × 3 coordinates
  │
  ▼
Root centering + normalization + 30-frame windows
  │
  ▼
Transformer pose refinement
  │
  ▼
Restore root position → floor alignment → temporal smoothing
  │
  ▼
CSV export ── frame, landmark, x, y, z, visibility
  │
  ▼
Unity character playback + camera / lighting
(course demonstration; Unity source not included here)
```

### What the project explores

- **Temporal refinement:** use neighboring poses rather than treating every frame independently.
- **Denoising:** reconstruct pose sequences after Gaussian-noise augmentation.
- **Grounding and smoothness:** combine floor calibration with Savitzky-Golay filtering.
- **Motion transfer:** export joint coordinates for character playback in Unity.
- **Cinematic presentation:** coordinate motion, camera tracking, and lighting in the final course demonstration.

## Model and representation

The main training entry point is [`src/train.py`](src/train.py), which defines its own sequence-to-sequence model.

| Component | Current training implementation |
| --- | --- |
| Input | `(batch, 30, 99)`: 30 frames, 33 joints, XYZ coordinates |
| Projection | Dense layer to 128 features, ReLU, dropout 0.2 |
| Attention | 4 heads; Keras `key_dim=128` per head |
| Feed-forward block | Two 128-unit dense layers with residual connections and layer normalization |
| Output | `(batch, 30, 99)`; inference uses the last predicted frame of each window |
| Objective | MSE reconstruction loss; MAE logged during training |
| Optimization | Adam, learning rate 0.001, batch size 64 |
| Training limit | Up to 50 epochs; early stopping with patience 5 |
| Augmentation | Gaussian noise with standard deviation 0.02 |

The targets in the current preprocessing path are copies of the extracted pose sequences, not independently measured motion-capture ground truth. The model learns reconstruction under added noise; this alone does not establish absolute 3D pose accuracy.

**Model variants are separate.** [`src/build_model.py`](src/build_model.py) defines a sequence-to-single-frame model with learned positional embeddings and output `(batch, 99)`. It is not imported by `src/train.py`. The training model does not explicitly add positional embeddings. Do not substitute these models without adapting the inference output handling.

## Motion processing

### Pose extraction and root motion

- MediaPipe world landmarks are exported with the Y coordinate negated.
- Hip landmarks 23 and 24 define the root used for centering.
- [`scripts/03_create_test_keypoints.py`](scripts/03_create_test_keypoints.py) adds a heuristic translation estimate using screen-space hip displacement and changes in hip width.
- Current translation sensitivities are 0.2 for X and 0.1 for Z.

This screen-space heuristic is not camera-motion compensation or calibrated global trajectory recovery. Camera movement, cuts, occlusion, and scale changes can affect the estimate.

### Refinement and export

[`src/test.py`](src/test.py) restores the original root after model inference, then applies:

1. **Floor calibration:** a sequence-level offset based on the first percentile of heel/toe Y coordinates.
2. **Initial smoothing:** Savitzky-Golay filter, window 9, polynomial order 2.
3. **Linear resampling:** source and target rates are currently both configured as 30 fps.
4. **Final smoothing:** Savitzky-Golay filter, window 15, polynomial order 2.

Floor calibration is not a per-frame contact lock or a physical foot-sliding constraint. Symmetric smoothing also uses future samples, so the full pipeline should not be described as a strictly causal real-time system.

## Repository layout

```text
.
├── environment.yml
├── scripts/
│   ├── 01_create_raw_keypoints.py      # Training-source video → NPZ rows
│   ├── 02_process_height_dataset.py   # Root centering, windows, normalization
│   └── 03_create_test_keypoints.py    # Test video → NPZ and *_raw.npy
├── src/
│   ├── train.py                      # Main sequence-to-sequence training path
│   ├── test.py                       # Inference, post-processing, CSV export
│   ├── build_model.py                # Separate sequence-to-frame model
│   ├── dataset.py                    # Alternative dataset-loading utility
│   └── compare_architecture.py       # Synthetic-data architecture experiment
├── utils/
│   └── viser_test.py                 # PoseViser visualization class
├── experiments/
│   ├── transformer_model/            # SavedModel artifact
│   ├── height_mlp_model/             # Additional saved model
│   └── scaler.pkl                    # Normalization statistics
└── data/
    ├── raw_keypoints/                # Extracted training-source NPZ files
    ├── test_keypoints/               # Test NPZ and NPY sequences
    ├── processed/                    # Generated training arrays
    ├── output/                       # Refined CSV outputs
    └── unity/                        # CSV prepared for the Unity demo
```

## Setup

```bash
git clone https://github.com/khk0606/Video2Unity-Transformer-Based-Motion-Refinement.git
cd Video2Unity-Transformer-Based-Motion-Refinement

conda env create -f environment.yml
conda activate ai_ml_final

# Imported by scripts, but not listed explicitly in environment.yml:
python -m pip install scikit-learn yt-dlp
```

The environment specifies Python 3.10, TensorFlow 2.15.0, MediaPipe 0.10.21, and OpenCV 4.11.0.86. Platform-specific TensorFlow installation may require adjustment.

The commands below are documented from source inspection; a fresh end-to-end run was not performed for this README update. Only load model, pickle, and object-array files from sources you trust.

## Using the existing inference path

Check that the matching model and scaler are present:

- `experiments/transformer_model/`
- `experiments/scaler.pkl`

Input files must be named `*_raw.npy` and contain an array shaped `(T, 33, 3)`, with at least 30 frames. The `.npz` row format is not read directly by inference.

```bash
# Run from the repository root.
python src/test.py
```

The script scans `data/test_keypoints/*_raw.npy` and writes:

```text
data/output/final_<input-name>_smooth.csv
```

Existing files with the same output name may be overwritten. The supplied `AAAraw.npy` does not match the `*_raw.npy` glob; filenames must follow the expected suffix.

CSV schema:

| Column | Meaning |
| --- | --- |
| `frame` | Zero-based output frame index |
| `landmark` | MediaPipe landmark index, 0–32 |
| `x, y, z` | Post-processed joint coordinates |
| `visibility` | Set to 1.0 by the exporter; not a refined confidence estimate |

The Unity importer must implement the skeleton mapping and coordinate conventions. Importing this CSV alone does not provide a complete rig-retargeting system.

## Preparing data and training

**Before running preprocessing:** restrict the input search to training videos and implement a video-level train/validation split. The current default is not suitable for claiming leakage-free evaluation.

1. Obtain videos you have permission to process. The extraction scripts contain example URLs that must be reviewed before use.
2. Convert training NPZ rows to dense `(T, 33, 3)` arrays named `*_raw.npy`. The training extractor writes NPZ, while the main preprocessing script expects NPY; this conversion is not connected automatically.
3. In `scripts/02_process_height_dataset.py`, restrict `search_path` to the intended training inputs. Its current recursive `data/**/*_raw.npy` search also includes test and generated-data directories.
4. Split by source video before creating windows and fit normalization statistics on the training partition only. Adapt the preprocessing/training split accordingly.
5. Back up the existing scaler and checkpoint before regenerating or retraining.

After addressing these prerequisites, the entry points are:

```bash
python scripts/02_process_height_dataset.py
python src/train.py
```

Preprocessing generates `data/processed/combined_raw.npy`, `combined_target.npy`, and `experiments/scaler.pkl`. Training writes checkpoints to `experiments/transformer_model/`. Keep each model paired with the scaler used for its training.

## Course demonstration and observations

The course report presents a K-pop dance animation in Unity with an intro, performance, and outro. It describes motion interpolation, camera tracking, and scripted lighting alongside the refined character motion.

Reported qualitative observations include smoother joint trajectories, less visible jitter, and more stable ground placement than the unrefined baseline. These are course-demonstration observations, not independently reproduced benchmark results. No percentage improvement or absolute pose-accuracy claim is made here.

[`src/compare_architecture.py`](src/compare_architecture.py) trains an MLP and a Transformer on random synthetic inputs and targets. Its loss curves are an architecture experiment, **not evidence of performance on real dance sequences**.

## Reproducibility notes

| Item | Report or intended design | Current repository |
| --- | --- | --- |
| Data isolation | Held-out source video | Preprocessing recursively scans `data/`; exclusion is not enforced |
| Validation | Generalization to unseen motion | Training slices the combined window array 90/10; adjacent windows can overlap across the boundary |
| Normalization | Training-only statistics for evaluation | Statistics are computed before the current train/validation split |
| Positional information | Positional encoding described | Present in the separate model file, absent from the main training model |
| Noise standard deviation | 0.01 in the report | 0.02 in the training generator |
| Ground contact | Dynamic floor alignment / foot locking described | Sequence-level percentile-based offset |
| Unity playback | C# interpolation and camera/lighting control | CSV artifacts included; Unity project and C# source absent |

The saved checkpoint's training provenance and held-out evaluation have not been revalidated here. Resolving these differences is necessary before reporting reproducible generalization metrics.

## Data and security

- Use video sources and derived assets according to their applicable permissions and terms.
- Do not commit browser cookies, tokens, or other credentials. Download scripts should use only your own local authentication, when required.
- The repository currently contains a cookie-file path; any real exposed session credentials should be invalidated and removed from repository history.
- A project-wide license is not currently included. Do not assume all code, media, and derived data share the same reuse permissions.

## 한국어 요약

단일 영상에서 추출한 3D 관절 좌표를 30프레임 단위의 Transformer로 정제하고, 스무딩과 바닥 정렬을 거쳐 Unity 캐릭터 애니메이션에 활용한 프로젝트입니다. Python 처리 코드와 모델·CSV가 포함되어 있으며, 보고서의 Unity 연출 소스는 포함되어 있지 않습니다. 현재 코드의 데이터 분리 및 모델 버전 차이를 위에 명시했으며, 정량 성능을 재현하려면 해당 사항을 먼저 정리해야 합니다.
