![](RADIANT%20-BEAM3.png)

# radiant_beam — Q1 codebase

Multimodal AI for glioma analysis. NVIDIA Academic Grant Program.

## Modules

| Module | Purpose |
|---|---|
| `schema.py` | Niche label space (IvyGAP), rung hierarchy, provenance manifest, Q1 audit |
| `ingest/opensrh.py` | OpenSRH two-channel SRH patch reader (verified byte format) |
| `render/falsecolor.py` | Beer-Lambert virtual H&E; wavelength sensitivity sweep |
| `models/detectors.py` | Swin-backboned Mask R-CNN and Mask2Former |
| `experiments/learning_curve.py` | Pre-registered sample-efficiency ablation |
| `holder/scaling.py` | Hölder/WTMM exponents, Koopman ladder, zoom-Lyapunov |
| `topology/persistence.py` | Alpha shapes, persistent homology, niche topology QC |
| `roi/heterogeneity.py` | Procedural-noise null model for ROI detection |
| `cli.py` | `ingest-opensrh`, `audit`, `render-srh`, `selftest` |

## Quick start

```bash
python -m radiant_beam.cli selftest
python -m radiant_beam.cli ingest-opensrh /data/opensrh --pilot
python -m radiant_beam.cli audit
python -m radiant_beam.cli render-srh patch.tif --out he.png
```

## Dependencies

Core: numpy, scipy, scikit-image, tifffile, pydicom
Math:  PyWavelets (Hölder), gudhi (topology), opensimplex / noise (ROI)
Deep:  torch, torchvision, transformers, torch_geometric
Domain: ramanspy

Modules raise `ImportError` with an actionable message rather than degrading
silently when an optional dependency is absent.
