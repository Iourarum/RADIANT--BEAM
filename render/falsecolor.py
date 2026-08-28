"""
radiant_beam.render.falsecolor
==============================

Deterministic two-channel -> virtual H&E rendering.

This is the *algorithmic* conversion path, chosen over a learned/GAN approach
for the current project phase for two reasons:

1.  Data. Deep virtual-staining methods need large paired corpora and are
    documented as sensitive to pre-analytical variation (staining protocol,
    acquisition settings). We do not yet have co-registered pairs at volume.

2.  Auditability. A fixed formula cannot hallucinate tissue structure that is
    not in the input -- it can only recombine what was measured. A GAN
    optimises for "looks like real H&E", which is a different objective from
    "faithfully represents this tissue's chemistry". For anything that may
    later touch diagnostic interpretation, a deterministic and traceable
    mapping is the defensible default.

The model implemented here follows the Beer-Lambert absorption formulation used
by FalseColor-Python (Serafin et al.), which itself descends from the Gareau
(2009) additive model as refined by Bini et al. (2011) against real transmitted
stain spectra. Two pieces matter and are implemented separately:

    ``level_intensity``  -- flat-field-like local + global normalisation, which
                            is what makes the output stable across specimens
                            with uneven signal. Skipping this is the main
                            reason naive false-colouring looks inconsistent.

    ``beer_lambert_he``  -- the actual colour mapping.

Both are pure NumPy and side-effect free, so they are trivially testable and
run identically on workstation and Jetson.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

# Beer-Lambert absorbance coefficients, (R, G, B).
# These reproduce the conventional H&E palette: haematoxylin -> blue/purple
# (absorbs red/green), eosin -> pink (absorbs green/blue).
HEMATOXYLIN_RGB = np.array([0.860, 1.000, 0.310], dtype=np.float32)
EOSIN_RGB = np.array([0.050, 1.000, 0.544], dtype=np.float32)


@dataclass
class FalseColorParams:
    """Tunables for the rendering.

    Defaults are a reasonable starting point for SRH-derived input, but
    ``k_nuclear`` / ``k_cyto`` should be re-fit per instrument -- they set the
    absorbance scaling and therefore overall contrast.
    """

    # Defaults re-fitted for SRH input. The values carried over from the
    # fluorescence virtual-staining literature (k_nuclear ~2.5, k_cyto ~0.9)
    # assume a dedicated nuclear dye whose dynamic range is much larger than
    # the eosin-analogue channel. SRH's two channels are far closer in mean
    # intensity (measured: 9971 vs 9978 on a representative OpenSRH patch), so
    # those defaults over-weight haematoxylin and render the whole field
    # purple. Re-fitting to k_nuclear=1.2 / k_cyto=2.4 restores a conventional
    # pink-stroma / blue-nuclei balance. Re-fit per instrument.
    k_nuclear: float = 1.2           # haematoxylin channel absorbance gain
    k_cyto: float = 2.4              # eosin channel absorbance gain
    background_percentile: float = 2.0
    leveling_block_px: int = 64      # cube size for the leveling map
    leveling_alpha: float = 1.0      # brightness constant (alpha in the paper)
    clip_percentile: float = 99.5


def _subtract_background(chan: np.ndarray, percentile: float) -> np.ndarray:
    """Uniform background subtraction, clamped at zero."""
    bg = np.percentile(chan, percentile)
    return np.clip(chan.astype(np.float32) - float(bg), 0.0, None)


def level_intensity(chan: np.ndarray,
                    block_px: int = 64,
                    alpha: float = 1.0,
                    eps: float = 1e-6) -> np.ndarray:
    """Local + global intensity leveling (flat-fielding analogue).

    Builds a coarse median map over ``block_px`` blocks, upsamples it by linear
    interpolation to full resolution, and divides. This is what suppresses
    intra-specimen intensity gradients (uneven illumination, tissue thickness
    variation) so that the same tissue type renders the same colour regardless
    of where in the slide it sits.

    Implemented with median rather than mean because the block statistic must
    not be dragged by a handful of specular/saturated pixels, which SRH data
    has plenty of.
    """
    chan = chan.astype(np.float32)
    h, w = chan.shape
    bh = max(1, h // max(1, block_px))
    bw = max(1, w // max(1, block_px))

    # Coarse median map via block reduction.
    trimmed = chan[: bh * block_px, : bw * block_px] if (bh * block_px <= h and
                                                         bw * block_px <= w) else chan
    th, tw = trimmed.shape
    bh = max(1, th // block_px)
    bw = max(1, tw // block_px)
    blocks = trimmed[: bh * block_px, : bw * block_px].reshape(
        bh, block_px, bw, block_px
    )
    coarse = np.median(blocks, axis=(1, 3)).astype(np.float32)

    # Guard: an all-dark block would drive the divisor to zero.
    coarse = np.maximum(coarse, np.percentile(coarse[coarse > 0], 5)
                        if np.any(coarse > 0) else eps)

    # Upsample to full resolution.
    zoom = (h / coarse.shape[0], w / coarse.shape[1])
    leveling_map = ndi.zoom(coarse, zoom, order=1)
    leveling_map = leveling_map[:h, :w]
    if leveling_map.shape != chan.shape:  # zoom rounding
        pad_h = h - leveling_map.shape[0]
        pad_w = w - leveling_map.shape[1]
        leveling_map = np.pad(leveling_map, ((0, max(0, pad_h)), (0, max(0, pad_w))),
                              mode="edge")[:h, :w]

    return chan / (alpha * leveling_map + eps)


def beer_lambert_he(nuclear: np.ndarray,
                    cyto: np.ndarray,
                    params: FalseColorParams | None = None) -> np.ndarray:
    """Map a nuclear-like and a cytoplasm-like channel to virtual H&E RGB.

    Parameters
    ----------
    nuclear
        Protein / nucleic-acid-weighted channel. For SRH this is the
        2930 cm-1 (CH3) frame; for fluorescence, the nuclear stain.
    cyto
        Lipid / stroma-weighted channel. For SRH this is 2845 cm-1 (CH2);
        for fluorescence, the eosin analogue.

    Returns
    -------
    uint8 RGB array, (H, W, 3).
    """
    p = params or FalseColorParams()

    n = _subtract_background(nuclear, p.background_percentile)
    c = _subtract_background(cyto, p.background_percentile)

    n = level_intensity(n, p.leveling_block_px, p.leveling_alpha)
    c = level_intensity(c, p.leveling_block_px, p.leveling_alpha)

    # Normalise to [0, 1] on a robust upper percentile.
    def _norm(x: np.ndarray) -> np.ndarray:
        hi = np.percentile(x, p.clip_percentile)
        return np.clip(x / (hi + 1e-6), 0.0, 1.0)

    n = _norm(n)
    c = _norm(c)

    # Beer-Lambert: transmitted = exp(-k * concentration * absorbance).
    out = np.exp(-(p.k_nuclear * n[..., None] * HEMATOXYLIN_RGB[None, None, :]))
    out = out * np.exp(-(p.k_cyto * c[..., None] * EOSIN_RGB[None, None, :]))

    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_srh_patch(ch2: np.ndarray, ch3: np.ndarray,
                     params: FalseColorParams | None = None) -> np.ndarray:
    """Convenience wrapper mapping SRH channel semantics onto the H&E model.

    CH3 (2930 cm-1, protein/nucleic acid) drives the haematoxylin channel;
    CH2 (2845 cm-1, lipid) drives eosin.
    """
    return beer_lambert_he(nuclear=ch3, cyto=ch2, params=params)


# --------------------------------------------------------------------------- #
# Wavelength-choice sensitivity
# --------------------------------------------------------------------------- #

def channel_pair_sensitivity(cube: np.ndarray,
                             wavenumbers: np.ndarray,
                             candidate_pairs: list[tuple[float, float]],
                             embed_fn,
                             ) -> dict[tuple[float, float], float]:
    """Measure how much the downstream embedding moves as the band pair changes.

    Rationale: OpenSRH's 2845 / 2930 cm-1 pair is inherited convention. For a
    hyperspectral rig we can *choose*. This function sweeps candidate pairs,
    renders each, embeds it, and reports the mean displacement of the embedding
    relative to the reference (first) pair.

    A pair with low sensitivity is a robust choice -- small instrument drift in
    band selection will not move the diagnosis. A pair with high sensitivity is
    a "relevant direction" in the renormalisation-group sense and should either
    be avoided or explicitly calibrated against.

    ``embed_fn`` takes an (H, W, 3) uint8 array and returns a 1D vector.
    """
    if not candidate_pairs:
        return {}

    def _band(target: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(wavenumbers - target)))
        return cube[..., idx]

    ref_lo, ref_hi = candidate_pairs[0]
    ref_vec = embed_fn(beer_lambert_he(_band(ref_hi), _band(ref_lo)))
    ref_vec = np.asarray(ref_vec, dtype=np.float64).ravel()
    ref_norm = np.linalg.norm(ref_vec) + 1e-12

    out: dict[tuple[float, float], float] = {}
    for lo, hi in candidate_pairs:
        vec = np.asarray(embed_fn(beer_lambert_he(_band(hi), _band(lo))),
                         dtype=np.float64).ravel()
        out[(lo, hi)] = float(np.linalg.norm(vec - ref_vec) / ref_norm)
    return out
