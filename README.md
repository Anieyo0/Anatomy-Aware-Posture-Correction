# Anatomy-Aware Deformable Registration for Posture Correction in Veterinary Thoracic Radiographs (Code)

This repository provides a reference implementation of the paper:

**Anatomy-Aware Deformable Registration for Posture Correction in Veterinary Thoracic Radiographs**

The core idea is to make learning-based deformable image registration focus on the **thoracic ROI** (and avoid spurious deformation in non-ROI regions such as limbs/background) using:

- **Spatial normalization** (preprocessing) to reduce gross pose variance,
- **Intra-subject Cartesian pairing** to avoid unrealistic cross-subject morphing,
- **ASP (Anatomical Saliency Prior)**: an annotation-free anatomical weighting map,
- **PAE (Progressive Anatomical Emphasis)**: a smooth schedule that increases ASP influence during training,
- A **backbone-agnostic loss design** (this repo includes a tutorial implementation based on VoxelMorph-diff).

---

## Repository layout

> Note: The current snapshot contains standalone demos (ASP/PAE/spatial normalization) and a tutorial folder that demonstrates the VoxelMorph-diff + ASP + PAE pipeline on sample data.

```
.
├── requirements.txt
├── asp.py                            # ASP builder + 3D visualization (CLI)
├── ASP/
│   ├── asp.py                        # same-role demo script (kept for convenience)
│   ├── samples_asp/                  # small example cohort (subject_01~03)
│   └── outputs_asp_demo/             # demo outputs (asp_map.npy, asp_map_3d.png)
├── PAE/
│   ├── pae.py                        # PAE schedule + demo plot generator
│   ├── samples_pae/                  # small example cohort (subject_01~03)
│   └── outputs_pae_demo/             # demo output image
├── spatial_normalization/
│   └── spatial_normalization.py      # spatial normalization pipeline (CLI)
└── tutorial/
    ├── reproduce_voxmorph_diff_asp_pae.ipynb  # end-to-end tutorial (recommended entry)
    ├── sample_data/
    │   ├── images/
    │   │   ├── correct_posture/       # per-subject folders
    │   │   ├── incorrect_posture/     # per-subject folders
    │   │   └── asp_sources/           # hold-out correct-posture images used to build ASP
    │   └── pairs/                     # example pairing list(s)
    ├── script/
    │   ├── train_voxelmorph.py        # training utilities used by the notebook
    │   └── plot_training_logs.py      # optional logging plotter
    └── voxelmorph/                    # lightweight VoxelMorph(-diff) implementation + anatomy-aware losses
        ├── networks.py
        ├── losses.py                  # includes ASP-weighted + edge-domain similarity terms
        ├── anatomical_map_generator.py
        └── ...
```

---

## Installation

### 1) Create an environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

For training (tutorial), you also need **PyTorch** appropriate to your CUDA / CPU setup.
Install it via the official PyTorch instructions for your platform.

---

## Quickstart (recommended)

### Option A — Run the end-to-end tutorial notebook

Open:

- `tutorial/reproduce_voxmorph_diff_asp_pae.ipynb`

This notebook demonstrates:
1) how to prepare sample data,
2) how to build ASP from hold-out correct-posture images,
3) how PAE ramps the ASP contribution during training,
4) how to train a VoxelMorph-diff-style model with anatomy-aware losses.

---

## Standalone components

### 1) Spatial normalization (CLI)

This script reproduces the sequential preprocessing pipeline:
**denoise → margin/aspect alignment → x-translation (centroid-based)**.

```bash
python spatial_normalization/spatial_normalization.py \
  --input_root  tutorial/sample_data/images \
  --output_root tutorial/sample_data/images_normalized
```

Key arguments:
- `--ratio_source`: optionally specify a directory used to compute the target aspect ratio
- `--denoise_thr`: intensity threshold for denoising (uint8 scale)
- additional parameters are available in `--help`

```bash
python spatial_normalization/spatial_normalization.py --help
```

---

### 2) Build ASP (Anatomical Saliency Prior) + export demo artifacts

`asp.py` builds an ASP map from a cohort directory and saves:
- `asp_map.npy`
- `asp_map_3d.png` (3D surface visualization)

Example using included samples:

```bash
python asp.py \
  --root_dir ASP/samples_asp \
  --out_dir ASP/outputs_asp_demo \
  --base_image ASP/samples_asp/subject_01/subject_01_sample.png \
  --w 256 --h 256 \
  --alpha 1.0 \
  --sigma 3.0
```

Outputs:
- `ASP/outputs_asp_demo/asp_map.npy`
- `ASP/outputs_asp_demo/asp_map_3d.png`

---

### 3) PAE (Progressive Anatomical Emphasis) demo

PAE provides a smooth schedule (tanh-shaped transition) that increases the ASP-weighted term
as training progresses.

Run the demo plot:

```bash
python PAE/pae.py
```

Output:
- `PAE/outputs_pae_demo/progressive_anatomical_emphasis.png`

---

## Data convention (tutorial)

The tutorial uses a **per-subject directory structure** to enable intra-subject pairing:

- `correct_posture/<subject_id>/*.png`
- `incorrect_posture/<subject_id>/*.png`
- `asp_sources/<subject_id>/*.png` (hold-out correct-posture images used only for ASP building)

A simple pairing list format is included under:
- `tutorial/sample_data/pairs/`

---

## Reproducibility notes

- The tutorial is designed to be self-contained with the provided `tutorial/sample_data`.
- The `tutorial/voxelmorph` directory includes:
  - anatomy-aware similarity losses (ASP-weighted),
  - optional edge-domain similarity terms,
  - training utilities invoked from the notebook.

---

## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@article{anatomy_aware_vet_thorax_registration,
  title   = {Anatomy-Aware Deformable Registration for Posture Correction in Veterinary Thoracic Radiographs},
  author  = {--},
  journal = {--},
  year    = {--}
}
```

> Replace the BibTeX fields above with the final publication metadata.

---

## License

Add an appropriate license file for your intended release (e.g., MIT, Apache-2.0) and update this section accordingly.