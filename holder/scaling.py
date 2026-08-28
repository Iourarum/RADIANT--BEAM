"""
radiant_beam.holder.scaling
===========================

Regularity and scale-dynamics instrumentation for the four-rung hierarchy.

Status: EXPLORATORY. The individual mathematical ingredients are established --
wavelet-transform-modulus-maxima estimation of Hoelder exponents, scale-space as
a diffusion flow, the Koopman/renormalisation-group correspondence, and
multifractal texture features in tumour imaging. Their *combination* here, as
rung-to-rung propagators over a niche hierarchy, is this project's own synthesis
and has no published precedent. Results must be reported as "proposed and under
validation", not as an applied standard technique.

Three things live here:

``holder_exponent_1d`` / ``holder_exponent_map``
    Local regularity. Low alpha = rough / near-singular. Two uses: segmentation
    boundaries are by definition low-alpha ridges (a mask boundary is a
    discontinuity), and Raman peaks are sharp near-singular features against a
    smooth baseline -- so thresholding alpha yields an unsupervised *spectral*
    mask, a sanity check on learned attention.

``KoopmanLadder``
    Fits linear operators A_k mapping rung k embeddings to rung k+1. This makes
    the vertical-binding loss a falsifiable claim (approximate scale-linearity)
    rather than an unmotivated regulariser.

``zoom_lyapunov``
    Whether fine-scale perturbations are damped (negative exponent, "irrelevant"
    in RG language -- coarse diagnosis is robust) or amplified (positive,
    "relevant" -- fine detail changes the call, which needs scrutiny).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pywt
    _PYWT = True
except ImportError:  # pragma: no cover
    _PYWT = False


# --------------------------------------------------------------------------- #
# Hoelder exponents
# --------------------------------------------------------------------------- #

def holder_exponent_1d(signal: np.ndarray,
                       scales: np.ndarray | None = None,
                       wavelet: str = "gaus1",
                       min_maxima: int = 3) -> np.ndarray:
    """Pointwise Hoelder exponent of a 1D signal via WTMM.

    The continuous wavelet transform of a signal with local exponent alpha at
    t0 scales as |W(a, t0)| ~ a^(alpha + 1/2). Tracking modulus maxima across
    scales and regressing log|W| on log(a) recovers alpha.

    For Raman: this is the natural "spectral mask" generator. Peaks are sharp
    (low alpha) against a smooth fluorescence baseline (high alpha), so
    ``alpha < threshold`` selects bands without any supervision at all.

    Returns an array of the same length as ``signal``; positions where the fit
    was underdetermined are NaN rather than silently zero.
    """
    if not _PYWT:
        raise ImportError("PyWavelets required: pip install PyWavelets")

    sig = np.asarray(signal, dtype=np.float64).ravel()
    n = sig.size
    if scales is None:
        top = max(2.0, n / 8.0)
        scales = np.geomspace(1.0, top, num=16)

    coeffs, _ = pywt.cwt(sig, scales, wavelet)
    mod = np.abs(coeffs)                     # (n_scales, n)
    log_a = np.log(np.asarray(scales, dtype=np.float64))

    alpha = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        col = mod[:, i]
        good = col > 0
        if good.sum() < min_maxima:
            continue
        slope, _ = np.polyfit(log_a[good], np.log(col[good]), 1)
        alpha[i] = slope - 0.5
    return alpha


def holder_exponent_map(image: np.ndarray,
                        scales: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
                        eps: float = 1e-12) -> np.ndarray:
    """2D local regularity map via multiscale Gaussian-gradient magnitude.

    A full 2D WTMM is expensive on gigapixel data; this uses the standard
    cheaper surrogate -- regress log gradient magnitude on log scale. Low alpha
    marks edges and texture; high alpha marks smooth regions.

    Consumed by:
      * ``radiant_beam.roi`` as one heterogeneity signal,
      * mask QC -- learned mask boundaries should coincide with low-alpha
        ridges; large disagreement is a flag on the mask, not a curiosity,
      * adaptive inference -- high alpha means intermediate zoom levels can be
        skipped and extrapolated, spending compute only where structure is
        rough. This is the "quickening" half of the scheme and is what makes it
        relevant to the sub-3-second Jetson target.
    """
    from scipy import ndimage as ndi

    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)

    logs, mags = [], []
    for s in scales:
        gy = ndi.gaussian_filter(img, sigma=s, order=(1, 0))
        gx = ndi.gaussian_filter(img, sigma=s, order=(0, 1))
        mags.append(np.sqrt(gx ** 2 + gy ** 2) + eps)
        logs.append(np.log(s))

    log_a = np.asarray(logs)
    stack = np.log(np.stack(mags, axis=0))

    # Vectorised per-pixel least squares.
    a_mean = log_a.mean()
    a_dev = log_a - a_mean
    denom = float(np.sum(a_dev ** 2)) + eps
    y_mean = stack.mean(axis=0)
    num = np.tensordot(a_dev, stack - y_mean, axes=(0, 0))
    return num / denom


def scaling_fit_quality(embeddings_by_rung: dict[int, np.ndarray],
                        rung_scales_um: dict[int, float]) -> dict[str, float]:
    """Test whether embedding displacement follows a power law across rungs.

    Fits log||delta phi|| ~ alpha * log(delta s). A poor R^2 means the
    vertical-binding assumption is breaking down for this region -- reportable
    *before* and independently of any downstream accuracy metric, which makes it
    a genuine QC signal rather than a post-hoc rationalisation.
    """
    rungs = sorted(set(embeddings_by_rung) & set(rung_scales_um))
    if len(rungs) < 3:
        return {"alpha": float("nan"), "r2": float("nan"), "n_points": len(rungs)}

    d_log_s, d_log_phi = [], []
    for a, b in zip(rungs[:-1], rungs[1:]):
        sa, sb = rung_scales_um[a], rung_scales_um[b]
        if not (sa > 0 and sb > 0):
            continue
        pa = np.asarray(embeddings_by_rung[a], dtype=np.float64).ravel()
        pb = np.asarray(embeddings_by_rung[b], dtype=np.float64).ravel()
        k = min(pa.size, pb.size)
        disp = np.linalg.norm(pb[:k] - pa[:k])
        if disp <= 0:
            continue
        d_log_s.append(abs(np.log(sb) - np.log(sa)))
        d_log_phi.append(np.log(disp))

    if len(d_log_s) < 2:
        return {"alpha": float("nan"), "r2": float("nan"), "n_points": len(d_log_s)}

    x = np.log(np.asarray(d_log_s))
    y = np.asarray(d_log_phi)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": float(slope), "r2": float(r2), "n_points": len(d_log_s)}


# --------------------------------------------------------------------------- #
# Koopman ladder over the zoom axis
# --------------------------------------------------------------------------- #

@dataclass
class KoopmanRung:
    from_rung: int
    to_rung: int
    operator: np.ndarray
    residual: float
    condition_number: float


class KoopmanLadder:
    """Linear propagators between adjacent rungs, fitted by EDMD-style least
    squares: phi(x, s_{k+1}) ~= A_k phi(x, s_k).

    This *is* vertical binding, made precise. The residual is exactly the
    vertical-binding loss; the difference is that the operator is explicit and
    inspectable, so a failure can be attributed to a particular rung transition
    rather than showing up as an undifferentiated loss value.
    """

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge
        self.rungs: list[KoopmanRung] = []

    def fit(self, embeddings_by_rung: dict[int, np.ndarray]) -> "KoopmanLadder":
        """``embeddings_by_rung[k]`` is (n_samples, dim_k)."""
        self.rungs = []
        keys = sorted(embeddings_by_rung)
        for a, b in zip(keys[:-1], keys[1:]):
            X = np.asarray(embeddings_by_rung[a], dtype=np.float64)
            Y = np.asarray(embeddings_by_rung[b], dtype=np.float64)
            if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
                raise ValueError(
                    f"rung {a}->{b}: need matching sample counts, "
                    f"got {X.shape} and {Y.shape}"
                )
            # Ridge-regularised least squares: A = Y^T X (X^T X + lI)^-1
            G = X.T @ X + self.ridge * np.eye(X.shape[1])
            A = np.linalg.solve(G, X.T @ Y).T

            resid = float(np.linalg.norm(Y - X @ A.T) / (np.linalg.norm(Y) + 1e-12))
            cond = float(np.linalg.cond(A)) if A.size else float("nan")
            self.rungs.append(KoopmanRung(a, b, A, resid, cond))
        return self

    def zoom_lyapunov(self, rung_scales_um: dict[int, float]) -> float:
        """Finite-scale Lyapunov exponent across the whole ladder.

        lambda = log||prod A_k|| / (s_fine - s_coarse), in log-scale units.

        Negative: fine-scale perturbations are damped as we coarsen -- the
        slide/niche-level call is robust to fine noise. ("Irrelevant direction".)

        Positive: fine detail propagates up and can change the coarse call.
        Either the model is tracking real fine-grained heterogeneity -- entirely
        plausible in glioma -- or it is keying on a nuisance parameter. Resolve
        with the deletion/insertion and Grad-CAM checks; the exponent alone does
        not distinguish those two cases.
        """
        if not self.rungs:
            return float("nan")

        prod = np.eye(self.rungs[0].operator.shape[1])
        for r in self.rungs:
            if r.operator.shape[1] != prod.shape[0]:
                return float("nan")   # dimension mismatch; ladder not composable
            prod = r.operator @ prod

        scales = [rung_scales_um.get(r.from_rung) for r in self.rungs]
        scales.append(rung_scales_um.get(self.rungs[-1].to_rung))
        scales = [s for s in scales if s and s > 0]
        if len(scales) < 2:
            return float("nan")

        span = abs(np.log(scales[0]) - np.log(scales[-1]))
        if span <= 0:
            return float("nan")
        return float(np.log(np.linalg.norm(prod, 2) + 1e-12) / span)

    def observability_gramian(self) -> np.ndarray:
        """Gramian W = sum_k (prod_{j<k} A_j)^T (prod_{j<k} A_j).

        Its conditioning says which directions of the underlying state are
        actually recoverable from the observed rungs. A poorly-conditioned
        Gramian means the embedding is discarding dynamically relevant structure
        even if static clustering metrics (Silhouette, Davies-Bouldin) look fine
        -- which is precisely the failure those metrics cannot see.
        """
        if not self.rungs:
            return np.zeros((0, 0))
        d = self.rungs[0].operator.shape[1]
        W = np.zeros((d, d))
        prod = np.eye(d)
        for r in self.rungs:
            W += prod.T @ prod
            if r.operator.shape[1] != prod.shape[0]:
                break
            prod = r.operator @ prod
        return W

    def report(self, rung_scales_um: dict[int, float]) -> dict[str, float | list]:
        W = self.observability_gramian()
        return {
            "n_transitions": len(self.rungs),
            "residuals": [r.residual for r in self.rungs],
            "mean_residual": float(np.mean([r.residual for r in self.rungs]))
            if self.rungs else float("nan"),
            "zoom_lyapunov": self.zoom_lyapunov(rung_scales_um),
            "gramian_condition": float(np.linalg.cond(W)) if W.size else float("nan"),
        }
