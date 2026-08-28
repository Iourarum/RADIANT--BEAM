"""
radiant_beam.schema
===================

Canonical label space and provenance tracking for RADIANT-BEAM.

Two things live here, and they are the load-bearing pieces of the whole
harmonisation story:

1.  ``NicheLabel`` -- the shared target space every modality-specific encoder
    must project onto. Based on the five IvyGAP anatomic structures. This is
    what turns "embedding unification" from an unfalsifiable claim into a
    measurable property: do two encoders, looking at the same physical region
    through different physics, agree on the niche?

2.  ``Provenance`` / ``SampleRecord`` -- per-sample source, licence, checksum
    and access tier. Without this, "1,000 GB curated" means nothing more than
    "1,000 GB downloaded".

The ``Rung`` enum encodes the four-level spatial hierarchy (slide -> niche ->
cell -> molecular) used for vertical/horizontal binding losses.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Label space
# --------------------------------------------------------------------------- #

class NicheLabel(IntEnum):
    """Canonical niche label space, after the IvyGAP anatomic structures.

    Integer values are stable and are what gets written into mask rasters.
    ``BACKGROUND`` is 0 so that an all-zero mask is a valid "nothing here".
    """

    BACKGROUND = 0
    LEADING_EDGE = 1
    INFILTRATING_TUMOR = 2
    CELLULAR_TUMOR = 3
    MICROVASCULAR_PROLIFERATION = 4
    PSEUDOPALISADING_NECROSIS = 5

    @classmethod
    def tumor_classes(cls) -> tuple["NicheLabel", ...]:
        """Everything that is not background."""
        return tuple(m for m in cls if m is not cls.BACKGROUND)

    @property
    def expects_loop_topology(self) -> bool:
        """Whether this niche is expected to show non-trivial H1 (loops).

        Pseudopalisading necrosis is, structurally, a ring of densely packed
        cells around a necrotic core -- i.e. a 1-cycle. Microvascular
        proliferation likewise produces vessel-lumen rings. This flag is
        consumed by ``radiant_beam.topology.persistence`` to decide whether a
        non-zero Betti-1 is corroborating evidence or an anomaly.
        """
        return self in (
            NicheLabel.PSEUDOPALISADING_NECROSIS,
            NicheLabel.MICROVASCULAR_PROLIFERATION,
        )


NICHE_COLORS: dict[NicheLabel, tuple[int, int, int]] = {
    NicheLabel.BACKGROUND: (0, 0, 0),
    NicheLabel.LEADING_EDGE: (66, 133, 244),
    NicheLabel.INFILTRATING_TUMOR: (52, 168, 83),
    NicheLabel.CELLULAR_TUMOR: (251, 188, 5),
    NicheLabel.MICROVASCULAR_PROLIFERATION: (234, 67, 53),
    NicheLabel.PSEUDOPALISADING_NECROSIS: (156, 39, 176),
}


class Rung(IntEnum):
    """The four-rung spatial hierarchy.

    Ordered coarse -> fine. ``SLIDE`` is rung 0. Vertical binding losses are
    computed between adjacent rungs; horizontal binding within a rung across
    modalities.
    """

    SLIDE = 0
    NICHE = 1
    CELL = 2
    MOLECULAR = 3

    @property
    def approx_micrometres(self) -> float | None:
        """Nominal physical scale, used as the log-scale axis for the Koopman /
        Hoelder analysis. ``MOLECULAR`` returns None -- it is not a spatial
        scale in the same sense and must not be fed to the scaling fit.
        """
        return {
            Rung.SLIDE: 10_000.0,
            Rung.NICHE: 500.0,
            Rung.CELL: 10.0,
            Rung.MOLECULAR: None,
        }[self]

    @classmethod
    def spatial_rungs(cls) -> tuple["Rung", ...]:
        """Rungs with a well-defined physical length scale."""
        return (cls.SLIDE, cls.NICHE, cls.CELL)


class Modality(str, Enum):
    RAMAN_POINT = "raman_point"          # 1D spectra, path A
    RAMAN_HSI = "raman_hsi"              # hyperspectral cube, path B
    SRH = "srh"                          # two-channel stimulated Raman histology
    FLUORESCENCE = "fluorescence"
    WSI = "wsi"
    PPI = "ppi"
    MASS_SPEC = "mass_spec"              # validation-only, not fused

    @property
    def is_fused(self) -> bool:
        """Whether this modality enters the unified embedding.

        Mass spectrometry is deliberately excluded: it is used as an offline
        validation and calibration signal for the BioID interactome and the
        interpretability layer, not as a fifth alignment problem.
        """
        return self is not Modality.MASS_SPEC


class AccessTier(str, Enum):
    PUBLIC = "public"                    # redistributable
    REGISTERED = "registered"            # requires account / DUA
    INTERNAL = "internal"                # in-house, not redistributable
    SYNTHETIC = "synthetic"


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

@dataclass
class Provenance:
    """Where a sample came from and what we may do with it."""

    source: str                          # e.g. "OpenSRH", "IDC", "HPA"
    source_id: str                       # dataset-native identifier
    licence: str                         # e.g. "CC-BY-4.0", "MIT", "internal"
    access_tier: AccessTier
    retrieved_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    checksum_sha256: str | None = None
    source_url: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["access_tier"] = self.access_tier.value
        return d


@dataclass
class SampleRecord:
    """One curated sample, whatever the modality."""

    sample_id: str
    modality: Modality
    rung: Rung
    path: str
    provenance: Provenance
    niche_label: NicheLabel | None = None
    patient_id: str | None = None
    slide_id: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    size_bytes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "modality": self.modality.value,
            "rung": int(self.rung),
            "path": self.path,
            "provenance": self.provenance.to_dict(),
            "niche_label": int(self.niche_label) if self.niche_label is not None else None,
            "patient_id": self.patient_id,
            "slide_id": self.slide_id,
            "shape": list(self.shape) if self.shape else None,
            "dtype": self.dtype,
            "size_bytes": self.size_bytes,
            "extra": self.extra,
        }


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 so we can checksum gigapixel WSI without loading it."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class Manifest:
    """Append-only JSONL manifest of curated samples.

    JSONL rather than a single JSON array so that ingestion can be interrupted
    and resumed without rewriting, and so that two ingest workers can append
    concurrently without a lock.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: SampleRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")

    def extend(self, records: Iterable[SampleRecord]) -> int:
        n = 0
        with open(self.path, "a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.to_dict(), separators=(",", ":")) + "\n")
                n += 1
        return n

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    # -- reporting helpers -------------------------------------------------- #

    def counts_by_modality(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.read():
            out[rec["modality"]] = out.get(rec["modality"], 0) + 1
        return out

    def bytes_by_modality(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.read():
            sz = rec.get("size_bytes") or 0
            out[rec["modality"]] = out.get(rec["modality"], 0) + sz
        return out

    def q1_target_report(self, per_modality_target: int = 500,
                         total_bytes_target: int = 1_000 * (1 << 30)) -> dict[str, Any]:
        """Direct answer to the two Q1 numeric commitments.

        Returns per-modality counts against the >=500 target and total curated
        volume against the ~1,000 GB target. Deliberately reports the two
        separately: hitting the GB number by over-pulling WSI while
        under-resourcing fluorescence/PPI would satisfy the letter of the
        commitment while leaving the multimodal dataset unbalanced.
        """
        counts = self.counts_by_modality()
        sizes = self.bytes_by_modality()
        total = sum(sizes.values())
        return {
            "per_modality": {
                m: {
                    "count": c,
                    "target": per_modality_target,
                    "met": c >= per_modality_target,
                    "bytes": sizes.get(m, 0),
                }
                for m, c in sorted(counts.items())
            },
            "total_bytes": total,
            "total_gb": round(total / (1 << 30), 2),
            "total_bytes_target": total_bytes_target,
            "volume_target_met": total >= total_bytes_target,
            "modalities_meeting_count_target": sum(
                1 for c in counts.values() if c >= per_modality_target
            ),
        }
