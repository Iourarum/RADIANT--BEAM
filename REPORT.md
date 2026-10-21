# RADIANT-BEAM: first run on real OpenSRH images (2026-10-20)

## Data actually used
- The raw OpenSRH TIFFs need a data-request form (opensrh.mlins.org → "Request Data"). The FastGlioma checkpoints (`mlinslab/fastglioma` on Hugging Face) are manually gated. Neither was available here.
- What *is* public: the OpenSRH image viewer serves 7 full specimen mosaics as Deep Zoom tiles (PNG, 300 px tiles, top level = native resolution):
  NIO_001.4 hgg, NIO_064.1 lgg, NIO_131.5 mening, NIO_200.1 met, NIO_245.2 pit, NIO_271.3 schwan, NIO_302.2 normal.
  `scripts/fetch_dzi.py` stitches them (≈1.5 min). These are **device-rendered virtual H&E (RGB)**, not raw two-channel SRH.
- Route into the package: tile into 300×300 patches (2,510 in total), then `SRHColorModel.invert` (DEVICE_FIT_NIO003) to get *estimated* CH2/CH3. These are written as OpenSRH-format 2-page uint16 TIFFs, and everything downstream runs on them. **Every channel value below is an estimate from inversion, not a measurement.**
- Compute: 4-core CPU, no GPU.

## Codebase state
- `pytest`: 65 passed, 3 skipped (the skips are the NIO_003 matched-pair tests, because the pair isn't in the zip). Runtime 84 s on CPU.
- `selftest`: torchvision, transformers, nvflare, onnx, ramanspy, torch_geometric, numba and noise are missing. So detectors (Mask R-CNN / Mask2Former), FLARE and ONNX export can't run here.

## Results

### 1. Device colour model across 7 new specimens (stage1.json)
Round-trip render(invert(rgb)) vs. original:

| specimen | MAE (grey) | R² | legacy falsecolor MAE |
|---|---|---|---|
| hgg | 1.2 | 0.996 | 44.6 |
| lgg | 1.2 | 0.996 | 47.9 |
| mening | 1.5 | 0.995 | 61.2 |
| pit | 1.5 | 0.994 | 53.7 |
| **met** | **15.6** | **0.875** | 47.8 |
| **schwan** | **12.5** | **0.870** | 77.6 |
| **normal** | **12.8** | **0.917** | 45.7 |

- There are **two rendering regimes.** For 4 mosaics, every pixel lies on the NIO_003 two-stain manifold. For met, schwannoma and normal it does not: their nuclei are blue/purple, which the fitted stains can't produce, so the re-render shows them grey (see pipeline_panels.png). This is most likely a different device or software colour pipeline.
- In regime B the inverted channels are biased (CH3 < CH2 on average, 8.6–9.1k vs ~10k), while in regime A CH3 ≳ CH2.
- A near-perfect round trip only shows the RGB lies on the model's manifold. It doesn't validate the recovered channels physically; that still needs matched raw pairs.
- The legacy `render_srh_patch` is 45–78 grey levels off on every specimen, which confirms the README's warning about it.
- Macenko: the spread of mean RGB between specimens went from SD (16.6, 13.8, 27.8) to (13.5, 14.1, 10.1). The blue channel drops strongly; red and green barely change.

### 2. Ingest / manifest
- `ingest.opensrh.ingest` read all 2,510 TIFFs in 6 s and parsed every filename (patient_id = NIO_xxx). `q1_target_report`: SRH count met (2,510 ≥ 500); volume 0.84 GB.
- `is_mostly_empty` rejected 0 patches. That is correct here, since no full tile is blank, but its threshold is relative to the uint16 full scale. On inverted data the metric is ~0.23, far above the 0.02 cut, so it would also need checking on real low-signal tiles.

### 3. Pseudo-label bootstrap, niche rung (bootstrap.json)
- Sources: alpha complex over intensity centroids, plus an Otsu threshold on the smoothed nuclear-density field (added here as the second niche-rung source). I ran 40 patches per class.
- Mean agreement was 0.25–0.44 and **0 of 280 patches were unanimous**, so the package's verdict was "Low agreement — do not train". The alpha mask covers 64–78 % of every patch, so at this alpha it is close to "all tissue" and doesn't separate niches.

### 4. Per-patch features and runtime
- Per class: nuclei per patch range 73 (lgg) to 167 (normal). Significant H1 loops fall from ~150 (hgg) to ~10 (normal).
- **`roi.score_image` takes 42 s per 300×300 patch without numba, and 0.28 s with it.** numba should be a core requirement, not optional.
- Stage 2 over 2,510 patches took 16 min on 4 cores.
- skimage warns that `regionprops.mean_intensity` is deprecated; it is used in `roi/heterogeneity.py:177`.

### 5. Classification: what you "get", and why not to believe it (stage3.json)
Balanced accuracy over 7 classes:

| protocol | intensity | morphology | all |
|---|---|---|---|
| random patch 5-fold (RF) | 0.935 | 0.947 | **0.984** |
| left/right-half spatial block (RF) | 0.888 | 0.919 | 0.965 |

- These numbers measure **specimen identity, not diagnosis**: there is exactly one patient per class.
- `leakage_report` correctly flags the random split: LEAKING, 7/7 patients shared.
- The colour regime is predictable from mean intensity alone with balanced accuracy 0.997. That is a device confound aligned with 3 of the 7 classes.
- An honest patient-level estimate is impossible with this data.

### 6. Bug found in `splits.grouped_split`
With 7 patients and 7 distinct diagnoses, stratification makes one bucket per patient. With n = 1, `round(0.7)` sends every bucket to train. The function returns **train = all, val = test = empty** and only adds a note, `WARNING: an empty partition was produced`. It doesn't raise. That contradicts the design note "splits.py raises rather than warns", and the gate is the only thing that would catch it. Suggested fix: raise `LeakageError` when any partition is empty, or when a stratum has fewer than 3 groups while `stratify=True`.

## What's needed next
1. File the OpenSRH data request and the FastGlioma Hugging Face access request. Raw TIFFs would remove the inversion step, and `test_srh_device` matched-pair tests would then run.
2. Fit a second `SRHColorModel` for the blue-nuclei regime (met/schwan/normal), or at least key the colour model on the regime. Otherwise every cross-site experiment learns the renderer.
3. Fix `grouped_split` (above) and make numba a core dependency.
4. When FastGlioma weights arrive: run `opensrh_three_channel` on these patches into its encoder. The input transform is already exact.

Files: `scripts/` (fetch_dzi, stage1_ingest, stage2_features, stage3_classify, figures), `out/` (stage1.json, features.json, bootstrap.json, stage3.json, pipeline_panels.png), `data/` (mosaics, 2,510 patch TIFFs, manifest.jsonl).
