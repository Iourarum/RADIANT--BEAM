"""
radiant_beam.models.detectors
=============================

Instance-segmentation heads for the *cell* rung, both mounted on a shared Swin
trunk.

Why both heads exist
--------------------
Mask2Former's query-based decoder is a set-prediction head descended from DETR.
It carries weaker built-in spatial inductive bias than a region-proposal
network, which is documented to translate into greater data hunger; controlled
benchmarks have found transformer segmentation architectures failing to beat
well-tuned CNN baselines in the low-hundreds-of-samples regime -- which is
squarely where Q1/Q2 sits.

Rather than settle that by assertion, ``radiant_beam.experiments.learning_curve``
trains both on nested subsets and measures the degradation slope. This module
just makes the two comparable: same backbone, same input pipeline, same
augmentation, so the ablation isolates the head rather than the tuning effort.

Note on backbone sharing: the Swin trunk here is intended to be the *same*
pretrained trunk used by the embedding pipeline. That is the practical argument
for Swin over a CNN backbone -- one trunk, two consumers -- and it is why the
loader exposes ``freeze_backbone`` and ``export_backbone_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False
    torch = None  # type: ignore
    nn = object  # type: ignore


HeadName = Literal["maskrcnn", "mask2former"]


@dataclass
class DetectorConfig:
    head: HeadName = "maskrcnn"
    num_classes: int = 6                      # 5 niche classes + background
    swin_variant: str = "microsoft/swin-tiny-patch4-window7-224"
    pretrained: bool = True
    freeze_backbone: bool = False
    image_size: int = 512
    # Mask R-CNN specifics
    trainable_backbone_layers: int = 3
    # Mask2Former specifics
    m2f_checkpoint: str = "facebook/mask2former-swin-tiny-coco-instance"

    def describe(self) -> str:
        return (
            f"{self.head} | backbone={self.swin_variant} | "
            f"classes={self.num_classes} | frozen={self.freeze_backbone}"
        )


def _require_torch() -> None:
    if not _TORCH:
        raise ImportError(
            "PyTorch is required to build detectors. "
            "Install torch/torchvision matching your CUDA version."
        )


# --------------------------------------------------------------------------- #
# Mask R-CNN with a Swin backbone
# --------------------------------------------------------------------------- #

def build_maskrcnn_swin(cfg: DetectorConfig):
    """Swin backbone + FPN + Mask R-CNN head.

    This is the configuration the original Swin paper itself used for COCO
    instance segmentation, and it keeps the region-proposal inductive bias that
    makes the head comparatively sample-efficient.

    Implementation route: torchvision's ``MaskRCNN`` accepts any backbone that
    exposes ``out_channels`` and returns an OrderedDict of feature maps, so we
    wrap a HuggingFace Swin in a small adapter rather than depending on
    mmdetection. (mmdetection ships ready Swin+Mask R-CNN configs and is the
    faster route if you already have it installed -- see ``notes`` in the
    returned metadata.)
    """
    _require_torch()
    from collections import OrderedDict

    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
    from transformers import SwinModel

    class SwinFPNBackbone(nn.Module):
        """Adapts HF Swin's hierarchical stages into an FPN for torchvision."""

        def __init__(self, variant: str, pretrained: bool, out_channels: int = 256):
            super().__init__()
            self.swin = (
                SwinModel.from_pretrained(variant)
                if pretrained else
                SwinModel(SwinModel.config_class.from_pretrained(variant))
            )
            # Swin emits 4 stages at strides 4, 8, 16, 32.
            dims = getattr(self.swin.config, "embed_dim", 96)
            depths = len(getattr(self.swin.config, "depths", [2, 2, 6, 2]))
            in_channels_list = [dims * (2 ** i) for i in range(depths)]
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list, out_channels=out_channels
            )
            self.out_channels = out_channels

        def forward(self, x):
            out = self.swin(x, output_hidden_states=True)
            # reshaped_hidden_states are (B, C, H, W) per stage
            feats = out.reshaped_hidden_states
            od = OrderedDict((str(i), f) for i, f in enumerate(feats))
            return self.fpn(od)

    backbone = SwinFPNBackbone(cfg.swin_variant, cfg.pretrained)

    if cfg.freeze_backbone:
        for p in backbone.swin.parameters():
            p.requires_grad = False

    anchor_gen = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    model = MaskRCNN(
        backbone,
        num_classes=cfg.num_classes,
        rpn_anchor_generator=anchor_gen,
        min_size=cfg.image_size,
        max_size=cfg.image_size,
    )
    return model


# --------------------------------------------------------------------------- #
# Mask2Former
# --------------------------------------------------------------------------- #

def build_mask2former(cfg: DetectorConfig):
    """Mask2Former with a Swin backbone, via HuggingFace.

    Kept in the codebase specifically as the comparison arm. If the
    learning-curve ablation shows it degrading faster at small n, that is a
    reportable result, not a failure -- and if it does not, the earlier
    architectural decision should be revisited on the evidence.
    """
    _require_torch()
    from transformers import (
        Mask2FormerConfig,
        Mask2FormerForUniversalSegmentation,
    )

    if cfg.pretrained:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            cfg.m2f_checkpoint,
            num_labels=cfg.num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        m2f_cfg = Mask2FormerConfig.from_pretrained(cfg.m2f_checkpoint)
        m2f_cfg.num_labels = cfg.num_classes
        model = Mask2FormerForUniversalSegmentation(m2f_cfg)

    if cfg.freeze_backbone:
        for name, p in model.named_parameters():
            if "pixel_level_module.encoder" in name:
                p.requires_grad = False

    return model


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def build_detector(cfg: DetectorConfig) -> tuple[Any, dict[str, Any]]:
    """Build either head and return it with descriptive metadata."""
    if cfg.head == "maskrcnn":
        model = build_maskrcnn_swin(cfg)
        notes = (
            "Region-proposal head. Stronger spatial prior; expected to be the "
            "more sample-efficient arm. mmdetection ships equivalent ready "
            "configs if preferred over this torchvision adapter."
        )
    elif cfg.head == "mask2former":
        model = build_mask2former(cfg)
        notes = (
            "Query-based set-prediction head. Higher ceiling at scale; the "
            "hypothesis under test is that it degrades faster at small n."
        )
    else:
        raise ValueError(f"unknown head: {cfg.head!r}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, {
        "head": cfg.head,
        "config": cfg.describe(),
        "params_total": n_params,
        "params_trainable": n_trainable,
        "notes": notes,
    }


def export_backbone_state(model, head: HeadName) -> dict:
    """Pull just the Swin trunk weights out, for reuse by the embedding pipeline.

    This is the concrete payoff of the shared-trunk design: a trunk fine-tuned
    during segmentation can initialise the embedding encoders, and vice versa.
    """
    _require_torch()
    prefix = "backbone.swin." if head == "maskrcnn" else "model.pixel_level_module.encoder."
    return {
        k[len(prefix):]: v
        for k, v in model.state_dict().items()
        if k.startswith(prefix)
    }
