"""
radiant_beam.roi.heterogeneity
==============================

Region-of-interest scoring by deviation from a procedural-noise null model.

The idea, stated plainly: smooth procedurally-generated noise is what
"structureless" tissue looks like statistically. Generate a Perlin/OpenSimplex
field matched in frequency content to the boring background, compute local
statistics over both it and the real image at matching scales, and score regions
by how far the real data departs from the null. Genuinely interesting structure
stands out precisely because it does *not* look like smooth noise.

Three complementary signals are provided rather than one, because they fail
differently:

``entropy_heterogeneity``  -- cheap, unglamorous local-entropy baseline. Run it
                              everywhere as a sanity check.
``noise_deviation_score``  -- multi-scale departure from the null model. More
                              principled, more expensive.
``topological_heterogeneity`` -- Betti-1 density. Biologically interpretable
                              (ties to the pseudopalisading signature) rather
                              than purely statistical.

Treat them as three flags to be read together, not as competing estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

try:
    from opensimplex import OpenSimplex
    _OPENSIMPLEX = True
except ImportError:  # pragma: no cover
    _OPENSIMPLEX = False

try:
    import noise as _perlin
    _PERLIN = True
except ImportError:  # pragma: no cover
    _PERLIN = False


# --------------------------------------------------------------------------- #
# Procedural field generation
# --------------------------------------------------------------------------- #

def simplex_field(shape: tuple[int, int],
                  scale: float = 32.0,
                  octaves: int = 4,
                  persistence: float = 0.5,
                  lacunarity: float = 2.0,
                  seed: int = 0) -> np.ndarray:
    """Fractal OpenSimplex field, normalised to [0, 1].

    Octave summation gives a controllable roughness spectrum: more octaves and
    higher persistence produce finer structure. That control is the point --
    it lets us generate synthetic tissue with *known* complexity and, via the
    threshold, known ground-truth topology.
    """
    if not _OPENSIMPLEX:
        raise ImportError("opensimplex required: pip install opensimplex")

    h, w = shape
    gen = OpenSimplex(seed=seed)
    field = np.zeros(shape, dtype=np.float64)

    amplitude, frequency, norm = 1.0, 1.0, 0.0
    for _ in range(octaves):
        ys = np.arange(h) * frequency / scale
        xs = np.arange(w) * frequency / scale
        layer = np.empty(shape, dtype=np.float64)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                layer[i, j] = gen.noise2(x, y)
        field += amplitude * layer
        norm += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    field /= max(norm, 1e-12)
    lo, hi = field.min(), field.max()
    return (field - lo) / max(hi - lo, 1e-12)


def perlin_field(shape: tuple[int, int],
                 scale: float = 32.0,
                 octaves: int = 4,
                 persistence: float = 0.5,
                 lacunarity: float = 2.0,
                 seed: int = 0) -> np.ndarray:
    """Perlin equivalent of ``simplex_field`` via the C-backed ``noise`` package."""
    if not _PERLIN:
        raise ImportError("noise required: pip install noise")

    h, w = shape
    out = np.empty(shape, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            out[i, j] = _perlin.pnoise2(
                j / scale, i / scale,
                octaves=octaves, persistence=persistence,
                lacunarity=lacunarity, base=seed,
            )
    lo, hi = out.min(), out.max()
    return (out - lo) / max(hi - lo, 1e-12)


def synthetic_tissue_mask(shape: tuple[int, int],
                          threshold: float = 0.55,
                          scale: float = 32.0,
                          octaves: int = 4,
                          seed: int = 0,
                          backend: str = "simplex") -> tuple[np.ndarray, dict]:
    """Synthetic tissue-like mask with ground-truth topology by construction.

    Returns ``(mask, metadata)`` where metadata records the generating
    parameters *and* the measured Betti numbers. Because the field and the
    threshold are known, the topology is not an estimate -- it is a controlled
    variable. That is what makes this usable as a benchmark: we can ask "which
    segmentation head degrades faster as ground-truth topological complexity
    rises", which is a sharper question than raw accuracy on real data.
    """
    field = (simplex_field(shape, scale, octaves, seed=seed) if backend == "simplex"
             else perlin_field(shape, scale, octaves, seed=seed))
    mask = field > threshold

    meta: dict = {
        "backend": backend, "scale": scale, "octaves": octaves,
        "threshold": threshold, "seed": seed,
        "coverage": float(mask.mean()),
    }
    try:
        from ..topology.persistence import persistence_from_image
        summary = persistence_from_image(mask.astype(np.float64))
        meta["betti_0"] = summary.betti_0
        meta["betti_1"] = summary.betti_1
        meta["n_significant_loops"] = summary.n_significant_loops
    except ImportError:
        meta["topology"] = "gudhi unavailable; Betti numbers not measured"

    return mask, meta


# --------------------------------------------------------------------------- #
# Heterogeneity scoring
# --------------------------------------------------------------------------- #

@dataclass
class ROIScore:
    entropy: np.ndarray
    noise_deviation: np.ndarray
    combined: np.ndarray
    threshold: float

    def regions(self, min_area: int = 64) -> list[dict]:
        """Connected components of the thresholded combined score."""
        from skimage.measure import label, regionprops

        binary = self.combined > self.threshold
        lab = label(binary)
        out = []
        for r in regionprops(lab, intensity_image=self.combined):
            if r.area < min_area:
                continue
            out.append({
                "bbox": tuple(int(v) for v in r.bbox),
                "area": int(r.area),
                "centroid": tuple(float(v) for v in r.centroid),
                "mean_score": float(r.mean_intensity),
            })
        return sorted(out, key=lambda d: d["mean_score"], reverse=True)


def entropy_heterogeneity(image: np.ndarray, radius: int = 7) -> np.ndarray:
    """Local Shannon entropy. The cheap baseline; always worth running."""
    from skimage.filters.rank import entropy as rank_entropy
    from skimage.morphology import disk
    from skimage.util import img_as_ubyte

    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    lo, hi = np.nanmin(img), np.nanmax(img)
    norm = (img - lo) / max(hi - lo, 1e-12)
    return rank_entropy(img_as_ubyte(norm), disk(radius)).astype(np.float64)


def noise_deviation_score(image: np.ndarray,
                          scales: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0),
                          n_null: int = 4,
                          seed: int = 0) -> np.ndarray:
    """Multi-scale departure from a matched procedural-noise null.

    For each scale we compare local variance in the real image against the
    distribution of local variance in null fields generated at the same scale.
    The score is a z-like statistic averaged over scales.

    Falls back gracefully to a pure multi-scale variance measure if no
    procedural-noise backend is installed, rather than failing the pipeline --
    but records nothing about that silently; callers should check
    ``noise_backend_available()``.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    lo, hi = np.nanmin(img), np.nanmax(img)
    img = (img - lo) / max(hi - lo, 1e-12)

    have_null = _OPENSIMPLEX or _PERLIN
    acc = np.zeros_like(img)

    for s in scales:
        local_mean = ndi.uniform_filter(img, size=int(2 * s + 1))
        local_var = ndi.uniform_filter(img ** 2, size=int(2 * s + 1)) - local_mean ** 2
        local_var = np.clip(local_var, 0.0, None)

        if have_null:
            null_vars = []
            for k in range(n_null):
                try:
                    nf = simplex_field(img.shape, scale=float(s) * 4.0,
                                       octaves=3, seed=seed + k)
                except ImportError:
                    nf = perlin_field(img.shape, scale=float(s) * 4.0,
                                      octaves=3, seed=seed + k)
                nm = ndi.uniform_filter(nf, size=int(2 * s + 1))
                nv = ndi.uniform_filter(nf ** 2, size=int(2 * s + 1)) - nm ** 2
                null_vars.append(np.clip(nv, 0.0, None))
            null_stack = np.stack(null_vars, axis=0)
            mu = null_stack.mean(axis=0)
            sd = null_stack.std(axis=0) + 1e-9
            acc += (local_var - mu) / sd
        else:
            acc += local_var / (local_var.std() + 1e-9)

    return acc / max(len(scales), 1)


def topological_heterogeneity(image: np.ndarray,
                              window: int = 64,
                              stride: int = 32) -> np.ndarray:
    """Sliding-window Betti-1 density.

    More loop-like structure at a given threshold means, near enough by
    definition, more architectural complexity than a smoothly connected
    homogeneous region. Coarse (window-resolution) by construction; intended as
    an interpretable corroborating signal, not a fine-grained map.
    """
    from ..topology.persistence import persistence_from_image

    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    h, w = img.shape
    out = np.zeros(((h - window) // stride + 1, (w - window) // stride + 1))

    for i, y in enumerate(range(0, h - window + 1, stride)):
        for j, x in enumerate(range(0, w - window + 1, stride)):
            tile = img[y:y + window, x:x + window]
            try:
                out[i, j] = persistence_from_image(tile).n_significant_loops
            except ImportError:
                return np.zeros_like(out)
    return out


def score_image(image: np.ndarray,
                entropy_radius: int = 7,
                percentile: float = 90.0) -> ROIScore:
    """Combine the cheap and the principled signal into one ROI map."""
    ent = entropy_heterogeneity(image, radius=entropy_radius)
    dev = noise_deviation_score(image)

    def _z(a: np.ndarray) -> np.ndarray:
        return (a - a.mean()) / (a.std() + 1e-9)

    combined = 0.5 * _z(ent) + 0.5 * _z(dev)
    return ROIScore(
        entropy=ent,
        noise_deviation=dev,
        combined=combined,
        threshold=float(np.percentile(combined, percentile)),
    )


def noise_backend_available() -> dict[str, bool]:
    return {"opensimplex": _OPENSIMPLEX, "noise": _PERLIN}
