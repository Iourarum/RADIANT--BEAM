"""
radiant_beam.experiments.learning_curve
=======================================

Pre-registered ablation: does Mask2Former's query decoder actually degrade
faster than Mask R-CNN's region-proposal head at Q1-scale sample sizes?

Design
------
Train both heads on nested subsets (10 / 25 / 50 / 100 % of the labelled pool),
several seeds per size, same backbone / augmentation / schedule for both. Report
mask AP with dispersion, plus the fitted degradation slope.

The slope is the actual quantity of interest, not the win/loss at any single
size. A single split at n=50 is noisy enough to flip the ranking by chance,
which is why ``n_seeds`` defaults to 3 and the reporter emits mean +/- std
rather than a bare number.

Metrics are declared up front in ``PRE_REGISTERED_METRICS`` and written into the
results file, so the eventual write-up cannot quietly become a post-hoc
justification of whichever arm happened to look better.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

PRE_REGISTERED_METRICS: tuple[str, ...] = (
    "mask_ap",          # COCO-style AP averaged over IoU .50:.95
    "mask_ap50",
    "mask_ap75",
    "dice",
    "latency_ms_per_image",
    "peak_vram_mb",
)

DEFAULT_FRACTIONS: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)


@dataclass
class LearningCurveConfig:
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS
    n_seeds: int = 3
    heads: tuple[str, ...] = ("maskrcnn", "mask2former")
    epochs: int = 50
    batch_size: int = 4
    lr: float = 1e-4
    out_dir: str = "results/learning_curve"
    # Honest-comparison guards
    identical_augmentation: bool = True
    identical_schedule: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    head: str
    fraction: float
    n_train: int
    seed: int
    metrics: dict[str, float]
    wall_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


def subsample_indices(n_total: int, fraction: float, seed: int) -> np.ndarray:
    """Nested random subset.

    Nested (rather than independent) so that the smaller sets are subsets of the
    larger ones for a given seed -- this removes one source of variance from the
    curve, since a difference between adjacent points then reflects the added
    data rather than a wholly different draw.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    k = max(1, int(round(n_total * fraction)))
    return np.sort(perm[:k])


def fit_degradation_slope(fractions: Sequence[float],
                          scores: Sequence[float]) -> dict[str, float]:
    """Fit log(score) ~ a + b * log(fraction).

    ``b`` is the degradation exponent: larger b means performance falls off
    faster as data is removed, i.e. greater data hunger. Reporting it in log-log
    space makes the two heads comparable even when their absolute AP differs.
    """
    f = np.asarray(fractions, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    mask = (f > 0) & (s > 0)
    if mask.sum() < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}

    x = np.log(f[mask])
    y = np.log(s[mask])
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(b), "intercept": float(a), "r2": float(r2)}


def run(
    train_fn: Callable[..., dict[str, float]],
    n_total: int,
    cfg: LearningCurveConfig | None = None,
) -> list[RunResult]:
    """Execute the sweep.

    ``train_fn(head, indices, seed, cfg) -> metrics dict`` is injected so this
    module stays free of any particular training loop; the harness is about the
    experimental design, not the optimiser.
    """
    cfg = cfg or LearningCurveConfig()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write the pre-registration before any training happens.
    (out_dir / "preregistration.json").write_text(
        json.dumps(
            {
                "metrics": list(PRE_REGISTERED_METRICS),
                "config": cfg.to_dict(),
                "hypothesis": (
                    "Mask2Former (query decoder) shows a steeper log-log "
                    "degradation slope than Mask R-CNN (region proposal) as "
                    "training fraction decreases."
                ),
                "declared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results: list[RunResult] = []
    for head in cfg.heads:
        for frac in cfg.fractions:
            for seed in range(cfg.n_seeds):
                idx = subsample_indices(n_total, frac, seed)
                t0 = time.time()
                metrics = train_fn(head=head, indices=idx, seed=seed, cfg=cfg)
                dt = time.time() - t0

                missing = set(PRE_REGISTERED_METRICS) - set(metrics)
                results.append(
                    RunResult(
                        head=head,
                        fraction=frac,
                        n_train=len(idx),
                        seed=seed,
                        metrics={k: float(v) for k, v in metrics.items()},
                        wall_seconds=dt,
                        extra={"missing_metrics": sorted(missing)} if missing else {},
                    )
                )

    (out_dir / "raw_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    return results


def summarise(results: list[RunResult],
              metric: str = "mask_ap") -> dict[str, Any]:
    """Aggregate across seeds and fit the degradation slope per head."""
    by_head: dict[str, dict[float, list[float]]] = {}
    for r in results:
        if metric not in r.metrics:
            continue
        by_head.setdefault(r.head, {}).setdefault(r.fraction, []).append(
            r.metrics[metric]
        )

    summary: dict[str, Any] = {"metric": metric, "heads": {}}
    for head, per_frac in by_head.items():
        fracs = sorted(per_frac)
        means = [float(np.mean(per_frac[f])) for f in fracs]
        stds = [float(np.std(per_frac[f], ddof=1)) if len(per_frac[f]) > 1 else 0.0
                for f in fracs]
        summary["heads"][head] = {
            "fractions": fracs,
            "mean": means,
            "std": stds,
            "n_seeds": [len(per_frac[f]) for f in fracs],
            "degradation": fit_degradation_slope(fracs, means),
        }

    heads = list(summary["heads"])
    if len(heads) == 2:
        a, b = heads
        sa = summary["heads"][a]["degradation"]["slope"]
        sb = summary["heads"][b]["degradation"]["slope"]
        if np.isfinite(sa) and np.isfinite(sb):
            steeper = a if sa > sb else b
            summary["verdict"] = {
                "steeper_degradation": steeper,
                "slope_difference": abs(float(sa - sb)),
                "caveat": (
                    "Interpret only alongside the per-point std; overlapping "
                    "error bars mean the ranking is not established."
                ),
            }
    return summary
