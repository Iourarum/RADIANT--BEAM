"""
radiant_beam.topology.persistence
=================================

Topological instrumentation: alpha complexes and persistent homology.

Two distinct jobs, unified by the fact that GUDHI implements both on the same
Delaunay/alpha machinery:

1.  **Training-free mask generation.** Alpha shapes build a concave boundary
    around a point cloud (nuclei centroids from Cellpose/StarDist, or clustered
    embedding points). No training data required, which makes this a genuine
    third pseudo-label source alongside the pretrained segmenters during the
    label-scarce Q1 phase.

2.  **Niche topology signatures.** Pseudopalisading necrosis is structurally a
    ring of cells around a necrotic core -- a 1-cycle, i.e. non-zero Betti-1.
    Microvascular proliferation likewise yields vessel-lumen rings. So H1
    persistence gives an interpretable, architecture-grounded signature for
    exactly those two IvyGAP categories, rather than relying wholly on a learned
    classifier to notice the pattern.

Precedent: persistent homology has been applied to quantify necrosis in
glioblastoma tissue, and Betti-number features from 3D MRI have been used to
separate high- from low-grade glioma. The application to *niche-level masks* in
a multimodal hierarchy is this project's extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import gudhi
    _GUDHI = True
except ImportError:  # pragma: no cover
    _GUDHI = False


def _require_gudhi() -> None:
    if not _GUDHI:
        raise ImportError("GUDHI required: pip install gudhi")


# --------------------------------------------------------------------------- #
# Alpha shapes
# --------------------------------------------------------------------------- #

@dataclass
class AlphaShapeResult:
    alpha: float
    n_points: int
    n_simplices: int
    boundary_edges: list[tuple[int, int]]
    n_components: int


def alpha_shape(points: np.ndarray, alpha: float) -> AlphaShapeResult:
    """Alpha complex of a 2D/3D point cloud at a fixed alpha.

    Small alpha hugs the points tightly and may fragment into disconnected
    pieces or reveal cavities; large alpha approaches the convex hull. This is
    exactly the molecular-surface / cavity-detection machinery from MD, applied
    to nuclei centroids instead of atoms.
    """
    _require_gudhi()
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        raise ValueError(f"need >=3 points in an (n, d) array, got {pts.shape}")

    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree(max_alpha_square=float(alpha) ** 2)

    edges: list[tuple[int, int]] = []
    n_simplices = 0
    for simplex, _filt in st.get_filtration():
        n_simplices += 1
        if len(simplex) == 2:
            edges.append((int(simplex[0]), int(simplex[1])))

    st.compute_persistence(persistence_dim_max=True)
    betti = st.betti_numbers()

    return AlphaShapeResult(
        alpha=float(alpha),
        n_points=int(pts.shape[0]),
        n_simplices=n_simplices,
        boundary_edges=edges,
        n_components=int(betti[0]) if betti else 0,
    )


def alpha_shape_mask(points: np.ndarray,
                     shape: tuple[int, int],
                     alpha: float,
                     dilate: int = 1) -> np.ndarray:
    """Rasterise an alpha shape into a binary mask.

    Practical note: this produces a *region* mask, not instance masks. It is
    well suited to the niche rung (dense regions) and to bootstrapping
    coarse tissue-vs-background labels; it will not separate touching nuclei,
    which is what the learned instance heads are for.
    """
    from scipy import ndimage as ndi
    from skimage.draw import polygon as sk_polygon

    res = alpha_shape(points, alpha)
    pts = np.asarray(points, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)

    # Rasterise each alpha-complex triangle.
    _require_gudhi()
    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree(max_alpha_square=float(alpha) ** 2)
    for simplex, _ in st.get_filtration():
        if len(simplex) != 3:
            continue
        tri = pts[list(simplex)]
        rr, cc = sk_polygon(tri[:, 0], tri[:, 1], shape=shape)
        mask[rr, cc] = True

    if dilate > 0:
        mask = ndi.binary_dilation(mask, iterations=dilate)
    return mask


def suggest_alpha(points: np.ndarray, quantile: float = 0.9) -> float:
    """Heuristic alpha from nearest-neighbour spacing.

    Picking alpha by eye is the usual failure mode; this ties it to the actual
    point density so that the same code works on sparse and dense fields.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return 1.0
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    return float(np.quantile(d[:, 1], quantile) * 1.5)


# --------------------------------------------------------------------------- #
# Persistent homology
# --------------------------------------------------------------------------- #

@dataclass
class PersistenceSummary:
    betti_0: int
    betti_1: int
    total_persistence_h0: float
    total_persistence_h1: float
    max_persistence_h1: float
    n_significant_loops: int
    diagram: list[tuple[int, tuple[float, float]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "betti_0": self.betti_0,
            "betti_1": self.betti_1,
            "total_persistence_h0": self.total_persistence_h0,
            "total_persistence_h1": self.total_persistence_h1,
            "max_persistence_h1": self.max_persistence_h1,
            "n_significant_loops": self.n_significant_loops,
        }


def persistence_from_image(image: np.ndarray,
                           n_thresholds: int = 32,
                           significance: float = 0.05,
                           foreground_is_high: bool = True) -> PersistenceSummary:
    """Persistent homology of a scalar field.

    The threshold sweep *is* the filtration. A feature that survives a wide
    threshold range (large birth-death) is structural; one that appears and
    immediately dies is noise. This is the precise sense in which "big changes
    in topology" can be measured rather than eyeballed.

    Filtration direction
    --------------------
    ``foreground_is_high=True`` (the default) inverts the field so that bright
    structure enters the filtration first, and the reported Betti numbers
    therefore describe the *object*. This matters and is easy to get wrong: a
    naive sublevel filtration on a binary mask measures the topology of the
    **background**, which reports a spurious H1 for any solid blob (the
    background wraps around it) and misses the distinction we actually care
    about. Verified: a solid disc gives B0=1, B1=0 and an annulus gives
    B0=1, B1=1 under this convention, whereas the naive direction gives B1=1
    for both.

    Set ``foreground_is_high=False`` for fields where low values are the
    structure of interest (e.g. a Hoelder-exponent map, where singular ridges
    are minima).
    """
    _require_gudhi()
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)

    lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    rng = max(hi - lo, 1e-12)
    norm = (img - lo) / rng
    if foreground_is_high:
        norm = -norm

    cc = gudhi.CubicalComplex(
        dimensions=norm.shape, top_dimensional_cells=norm.flatten()
    )
    cc.compute_persistence()
    diag = cc.persistence()

    h0 = [(b, d) for dim, (b, d) in diag if dim == 0]
    h1 = [(b, d) for dim, (b, d) in diag if dim == 1]

    def _finite_persistence(pairs: Sequence[tuple[float, float]]) -> list[float]:
        return [
            (d - b) for b, d in pairs
            if np.isfinite(d) and np.isfinite(b) and d > b
        ]

    p0 = _finite_persistence(h0)
    p1 = _finite_persistence(h1)

    return PersistenceSummary(
        betti_0=len(h0),
        betti_1=len(h1),
        total_persistence_h0=float(sum(p0)),
        total_persistence_h1=float(sum(p1)),
        max_persistence_h1=float(max(p1)) if p1 else 0.0,
        n_significant_loops=int(sum(1 for p in p1 if p > significance)),
        diagram=[(int(dim), (float(b), float(d))) for dim, (b, d) in diag
                 if np.isfinite(d)],
    )


def niche_topology_consistency(mask: np.ndarray,
                               claimed_label: int,
                               significance: float = 0.05) -> dict[str, Any]:
    """Check a niche mask against its expected topology.

    Pseudopalisading necrosis and microvascular proliferation should show
    non-trivial H1. If a region is labelled as one of those but has no
    persistent loops, that is a flag worth surfacing -- either the mask is
    wrong, or the crop is too tight to contain the full ring.

    Deliberately returns a flag with a reason string rather than silently
    overriding the label: this is a QC signal for a human, not an auto-corrector.
    """
    from ..schema import NicheLabel

    summary = persistence_from_image(mask.astype(np.float64),
                                     significance=significance)
    try:
        label = NicheLabel(claimed_label)
    except ValueError:
        return {"ok": False, "reason": f"unknown niche label {claimed_label}",
                **summary.to_dict()}

    expects_loops = label.expects_loop_topology
    has_loops = summary.n_significant_loops > 0

    if expects_loops and not has_loops:
        reason = (
            f"{label.name} is expected to show ring topology (H1 > 0) but no "
            "persistent loops were found -- check mask quality or crop extent."
        )
        ok = False
    elif not expects_loops and summary.n_significant_loops > 3:
        reason = (
            f"{label.name} shows {summary.n_significant_loops} persistent loops, "
            "which is more structure than this niche class usually implies."
        )
        ok = False
    else:
        reason = "topology consistent with claimed niche label"
        ok = True

    return {"ok": ok, "reason": reason, "label": label.name,
            "expects_loops": expects_loops, **summary.to_dict()}
