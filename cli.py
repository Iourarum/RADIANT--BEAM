"""
radiant_beam.cli
================

Command-line entry points for Q1 operations.

The ``audit`` command is the one that matters for reporting: it reads the
curation manifest and emits exactly the numbers the Q1 commitment tables need
(per-modality sample counts against the >=500 target, curated volume against the
~1,000 GB target), in a form that can be pasted straight into the quarterly
report rather than re-counted by hand.

Usage
-----
    python -m radiant_beam.cli ingest-opensrh  /data/opensrh --pilot
    python -m radiant_beam.cli audit           --manifest data/manifest.jsonl
    python -m radiant_beam.cli render-srh      patch.tif --out patch_he.png
    python -m radiant_beam.cli selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MANIFEST = "data/manifest.jsonl"


def _cmd_ingest_opensrh(args: argparse.Namespace) -> int:
    from .ingest import opensrh
    from .schema import Manifest

    man = Manifest(args.manifest)
    n = opensrh.ingest(
        args.root, man,
        limit=args.limit, pilot=args.pilot,
        compute_checksums=not args.no_checksums,
    )
    print(f"ingested {n} OpenSRH patches -> {args.manifest}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from .schema import Manifest

    man = Manifest(args.manifest)
    report = man.q1_target_report(
        per_modality_target=args.per_modality_target,
        total_bytes_target=int(args.volume_target_gb * (1 << 30)),
    )

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("\nQ1 CURATION AUDIT")
    print("=" * 68)
    print(f"{'modality':<22}{'count':>9}{'target':>9}{'met':>7}{'GB':>10}")
    print("-" * 68)
    for mod, d in report["per_modality"].items():
        print(f"{mod:<22}{d['count']:>9}{d['target']:>9}"
              f"{'yes' if d['met'] else 'NO':>7}{d['bytes']/(1<<30):>10.2f}")
    print("-" * 68)
    print(f"{'TOTAL':<22}{'':>9}{'':>9}{'':>7}{report['total_gb']:>10.2f}")
    print()
    print(f"volume target      : {args.volume_target_gb:.0f} GB  "
          f"({'met' if report['volume_target_met'] else 'NOT met'})")
    print(f"modalities at count target: "
          f"{report['modalities_meeting_count_target']}")

    # The caveat that matters, printed rather than buried.
    if report["volume_target_met"] and report["modalities_meeting_count_target"] < 4:
        print()
        print("NOTE: volume target met but fewer than four modalities reached "
              "the per-modality count target. Volume is dominated by WSI; "
              "check that the multimodal balance is genuine and not an "
              "artefact of over-pulling one modality.")
    return 0


def _cmd_render_srh(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image

    from .ingest.opensrh import read_patch
    from .render.falsecolor import FalseColorParams, render_srh_patch

    patch = read_patch(args.path, strict_shape=not args.loose)
    rgb = render_srh_patch(
        patch.ch2, patch.ch3,
        FalseColorParams(k_nuclear=args.k_nuclear, k_cyto=args.k_cyto),
    )
    Image.fromarray(rgb).save(args.out)
    print(f"wrote {args.out}  shape={rgb.shape}  "
          f"src={patch.path.name}  specimen={patch.name.specimen if patch.name else '?'}")
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    """Verify which optional dependencies are present.

    Deliberately reports rather than installs: on a cluster the environment is
    usually managed, and silently pip-installing into it is worse than saying
    what is missing.
    """
    checks = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-image": "skimage",
        "tifffile": "tifffile",
        "pydicom": "pydicom",
        "PyWavelets (Hoelder/WTMM)": "pywt",
        "GUDHI (topology)": "gudhi",
        "opensimplex (ROI null model)": "opensimplex",
        "noise (Perlin)": "noise",
        "torch": "torch",
        "torchvision (Mask R-CNN)": "torchvision",
        "transformers (Swin, Mask2Former)": "transformers",
        "torch_geometric (PPI rung)": "torch_geometric",
        "ramanspy": "ramanspy",
    }
    missing = []
    print("\nRADIANT-BEAM environment self-test")
    print("=" * 52)
    for label, mod in checks.items():
        try:
            __import__(mod)
            print(f"  [ok]      {label}")
        except ImportError:
            print(f"  [MISSING] {label}")
            missing.append(label)
    print("=" * 52)
    if missing:
        print(f"{len(missing)} optional component(s) unavailable. "
              "Modules depending on them will raise ImportError with a clear "
              "message rather than degrading silently.")
    else:
        print("all components present.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="radiant_beam",
        description="RADIANT-BEAM Q1 operations",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ingest-opensrh", help="register OpenSRH patches")
    a.add_argument("root")
    a.add_argument("--manifest", default=DEFAULT_MANIFEST)
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--pilot", action="store_true",
                   help="cap at 600 patches (just over the >=500 commitment)")
    a.add_argument("--no-checksums", action="store_true",
                   help="skip SHA-256 (faster; weakens provenance)")
    a.set_defaults(func=_cmd_ingest_opensrh)

    b = sub.add_parser("audit", help="Q1 commitment audit from the manifest")
    b.add_argument("--manifest", default=DEFAULT_MANIFEST)
    b.add_argument("--per-modality-target", type=int, default=500)
    b.add_argument("--volume-target-gb", type=float, default=1000.0)
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=_cmd_audit)

    c = sub.add_parser("render-srh", help="two-channel SRH -> virtual H&E")
    c.add_argument("path")
    c.add_argument("--out", default="virtual_he.png")
    c.add_argument("--k-nuclear", type=float, default=2.5)
    c.add_argument("--k-cyto", type=float, default=0.9)
    c.add_argument("--loose", action="store_true",
                   help="accept non-300x300 tiles")
    c.set_defaults(func=_cmd_render_srh)

    d = sub.add_parser("selftest", help="check optional dependencies")
    d.set_defaults(func=_cmd_selftest)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
