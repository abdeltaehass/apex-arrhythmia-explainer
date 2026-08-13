"""Phase 25 — synthetic ECG augmentation for rare classes.

Generates class-conditional 12-lead recordings with a diffusion model and measures whether
training on them helps the rare classes without hurting the common ones.

    from src.synthesis import make_rare, train_diffusion, sample, assess

The parts:

- :mod:`~src.synthesis.rarity`    induce measurable rarity (PTB-XL's real rare labels have
                                  1-5 test positives — too few to score)
- :mod:`~src.synthesis.augment`   classical signal-space augmentation, the control to beat
- :mod:`~src.synthesis.diffusion` conditional 1D DDPM over ``(12, 1000)`` signals
- :mod:`~src.synthesis.quality`   memorization, diversity, and physiologic plausibility
- :mod:`~src.synthesis.ablation`  the five arms, and the bootstrap that keeps the
                                  comparison honest
"""

from src.synthesis.ablation import ARMS, TrainConfig, build_arm_data, evaluate
from src.synthesis.augment import AugmentConfig, augment, augment_batch
from src.synthesis.diffusion import DiffusionConfig, UNet1D, cosine_alphas, sample, train_diffusion
from src.synthesis.quality import QualityReport, assess, diversity, memorization, physiology
from src.synthesis.rarity import DEFAULT_TARGETS, RarityScenario, make_rare, oversample_indices

__all__ = [
    "ARMS", "DEFAULT_TARGETS", "AugmentConfig", "DiffusionConfig", "QualityReport",
    "RarityScenario", "TrainConfig", "UNet1D", "assess", "augment", "augment_batch",
    "build_arm_data", "cosine_alphas", "diversity", "evaluate", "make_rare", "memorization",
    "oversample_indices", "physiology", "sample", "train_diffusion",
]
