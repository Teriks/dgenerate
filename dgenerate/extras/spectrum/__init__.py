"""
Spectrum sampling acceleration.

https://github.com/hanjq17/Spectrum
"""

from dgenerate.extras.spectrum.runtime import (
    SpectrumSettings,
    SpectrumUnsupported,
    activate,
    active_settings,
    apply,
    pop,
    push,
    cfg_multiplier,
    resolve_settings,
    schedule_counts,
    schedule_for,
    warn_schedule,
)

__all__ = [
    'SpectrumSettings',
    'SpectrumUnsupported',
    'activate',
    'active_settings',
    'apply',
    'pop',
    'push',
    'cfg_multiplier',
    'resolve_settings',
    'schedule_counts',
    'schedule_for',
    'warn_schedule',
]
