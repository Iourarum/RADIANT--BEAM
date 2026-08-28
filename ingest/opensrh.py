"""
radiant_beam.ingest.opensrh
===========================

Reader for OpenSRH-format stimulated Raman histology patches.

Format notes -- these were established by inspecting an actual OpenSRH patch
rather than from documentation, so they are worth stating precisely:

    * TIFF, 2 pages / frames, 300 x 300 px, uint16 (``I;16``)
    * ``ImageDescription`` on page 0 carries ``{"shape": [2, 300, 300]}``
    * ``Software`` tag reads ``tifffile.py``
    * ``Compression = 1`` (raw), ``PhotometricInterpretation = 1`` (min-is-black)
    * Channel 0 = 2845 cm-1  (CH2 stretch, lipid-rich)
    * Channel 1 = 2930 cm-1  (CH3 stretch, protein / nucleic-acid-rich)

The two channels are *not* pre-combined. Any RGB rendering is downstream of
this reader -- see ``radiant_beam.render.falsecolor``.

Filename convention observed: ``NIO_003-3-5000_4000_600_600.tif`` parses as
``{specimen}-{slide}-{x}_{y}_{w}_{h}.tif``. This is inferred from the naming
pattern, not from a published spec, so ``parse_patch_name`` returns ``None``
on anything it cannot confidently decompose rather than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import tifffile
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "tifffile is required for OpenSRH ingest: pip install tifffile"
    ) from exc

from ..schema import (
    AccessTier,
    Manifest,
    Modality,
    Provenance,
    Rung,
    SampleRecord,
    sha256_file,
)

# Raman shifts, cm^-1. Fixed by the SRH instrument, not tunable per-sample.
CH2_SHIFT_CM1 = 2845.0
CH3_SHIFT_CM1 = 2930.0

EXPECTED_PATCH_SHAPE = (2, 300, 300)

_PATCH_RE = re.compile(
    r"^(?P<specimen>[A-Za-z0-9]+_\d+)"
    r"-(?P<slide>\d+)"
    r"-(?P<x>\d+)_(?P<y>\d+)_(?P<w>\d+)_(?P<h>\d+)$"
)


@dataclass(frozen=True)
class PatchName:
    specimen: str
    slide: int
    x: int
    y: int
    w: int
    h: int


def parse_patch_name(stem: str) -> PatchName | None:
    """Decompose an OpenSRH patch filename stem.

    Returns ``None`` rather than raising or guessing if the stem does not match
    the observed convention -- callers should treat that as "unknown position"
    and fall back to the file path as the identifier.
    """
    m = _PATCH_RE.match(stem)
    if not m:
        return None
    return PatchName(
        specimen=m["specimen"],
        slide=int(m["slide"]),
        x=int(m["x"]),
        y=int(m["y"]),
        w=int(m["w"]),
        h=int(m["h"]),
    )


class OpenSRHPatch:
    """A single two-channel SRH patch."""

    __slots__ = ("ch2", "ch3", "path", "name")

    def __init__(self, ch2: np.ndarray, ch3: np.ndarray, path: Path,
                 name: PatchName | None):
        self.ch2 = ch2      # 2845 cm-1, lipid
        self.ch3 = ch3      # 2930 cm-1, protein
        self.path = path
        self.name = name

    @property
    def shape(self) -> tuple[int, int]:
        return self.ch2.shape  # type: ignore[return-value]

    def stacked(self) -> np.ndarray:
        """(2, H, W) array in canonical channel order."""
        return np.stack([self.ch2, self.ch3], axis=0)

    def subtracted(self) -> np.ndarray:
        """CH3 - CH2, the classic SRH contrast.

        Computed in float32 and *not* clipped -- negative values are physically
        meaningful (lipid-dominant pixels) and clipping them here would throw
        away signal that the downstream renderer may want.
        """
        return self.ch3.astype(np.float32) - self.ch2.astype(np.float32)

    def is_mostly_empty(self, threshold: float = 0.02,
                        percentile: float = 99.0) -> bool:
        """Cheap background-patch rejector.

        OpenSRH mosaics contain a lot of off-tissue area. A patch whose bright
        percentile is close to its floor carries no tissue and should not be
        counted toward the >=500-sample commitment.
        """
        both = np.concatenate([self.ch2.ravel(), self.ch3.ravel()])
        lo = float(both.min())
        hi = float(np.percentile(both, percentile))
        full = float(np.iinfo(self.ch2.dtype).max) if np.issubdtype(
            self.ch2.dtype, np.integer) else float(both.max() or 1.0)
        return (hi - lo) / max(full, 1.0) < threshold


def read_patch(path: str | Path, *, strict_shape: bool = True) -> OpenSRHPatch:
    """Load one OpenSRH ``.tif`` patch.

    Parameters
    ----------
    strict_shape
        If True, raise on anything that is not (2, 300, 300). Set False when
        ingesting non-standard tiles (e.g. your own re-tiling of a mosaic).
    """
    path = Path(path)
    arr = tifffile.imread(str(path))

    if arr.ndim != 3 or arr.shape[0] != 2:
        raise ValueError(
            f"{path.name}: expected a 2-channel stack, got shape {arr.shape}. "
            "This does not look like an OpenSRH patch."
        )
    if strict_shape and tuple(arr.shape) != EXPECTED_PATCH_SHAPE:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_PATCH_SHAPE}, got {tuple(arr.shape)}. "
            "Pass strict_shape=False to accept."
        )

    return OpenSRHPatch(
        ch2=arr[0], ch3=arr[1], path=path, name=parse_patch_name(path.stem)
    )


def iter_patches(root: str | Path, *, pattern: str = "*.tif",
                 skip_empty: bool = True,
                 strict_shape: bool = True) -> Iterator[OpenSRHPatch]:
    """Walk a directory of OpenSRH patches.

    Malformed files are skipped with a warning rather than aborting the walk --
    a single bad file in a 4-million-patch tree should not kill an ingest run.
    """
    import warnings

    for p in sorted(Path(root).rglob(pattern)):
        try:
            patch = read_patch(p, strict_shape=strict_shape)
        except Exception as exc:  # noqa: BLE001 - deliberate: keep walking
            warnings.warn(f"skipping {p}: {exc}", RuntimeWarning, stacklevel=2)
            continue
        if skip_empty and patch.is_mostly_empty():
            continue
        yield patch


def ingest(root: str | Path,
           manifest: Manifest,
           *,
           limit: int | None = None,
           pilot: bool = False,
           licence: str = "see opensrh.mlins.org terms",
           compute_checksums: bool = True) -> int:
    """Register OpenSRH patches into the curation manifest.

    ``pilot=True`` caps at 600 patches -- just over the >=500-per-modality Q1
    commitment, enough to verify the pipeline end-to-end without pulling the
    full corpus.
    """
    if pilot and limit is None:
        limit = 600

    n = 0
    for patch in iter_patches(root):
        if limit is not None and n >= limit:
            break

        nm = patch.name
        prov = Provenance(
            source="OpenSRH",
            source_id=patch.path.stem,
            licence=licence,
            access_tier=AccessTier.REGISTERED,
            checksum_sha256=sha256_file(patch.path) if compute_checksums else None,
            source_url="https://opensrh.mlins.org",
            notes=(
                f"two-channel SRH; ch0={CH2_SHIFT_CM1} cm-1 (CH2/lipid), "
                f"ch1={CH3_SHIFT_CM1} cm-1 (CH3/protein)"
            ),
        )
        rec = SampleRecord(
            sample_id=f"opensrh/{patch.path.stem}",
            modality=Modality.SRH,
            rung=Rung.CELL,  # a 300x300 SRH patch resolves individual nuclei
            path=str(patch.path),
            provenance=prov,
            patient_id=nm.specimen if nm else None,
            slide_id=f"{nm.specimen}-{nm.slide}" if nm else None,
            shape=tuple(patch.stacked().shape),
            dtype=str(patch.ch2.dtype),
            size_bytes=patch.path.stat().st_size,
            extra=(
                {"x": nm.x, "y": nm.y, "w": nm.w, "h": nm.h} if nm else
                {"note": "filename did not match expected OpenSRH convention"}
            ),
        )
        manifest.append(rec)
        n += 1
    return n
