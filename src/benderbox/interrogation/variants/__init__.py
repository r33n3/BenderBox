"""
Jailbreak Variant Generator for BenderBox

Generates prompt variants using various jailbreak techniques
to test model robustness against adversarial inputs.
"""

from .generator import (
    VariantGenerator,
    VariantTechnique,
    VariantConfig,
    generate_variants,
)

__all__ = [
    "VariantGenerator",
    "VariantTechnique",
    "VariantConfig",
    "generate_variants",
]
