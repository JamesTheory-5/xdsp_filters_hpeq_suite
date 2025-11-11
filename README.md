# xdsp_filters_hpeq_suite

```python
#!/usr/bin/env python3
"""
xdsp_hpeq_suite.py

Audio-grade high-order EQ module.

Core design goals:
- Only RBJ-style biquads (Audio EQ Cookbook).
- High-order behavior via cascades:
    * Peaking EQ: Butterworth-derived Q distribution.
    * Low/High shelves: cascades of RBJ shelves.
    * Band shelves: pair of shelves.
- Stable, monotonic, no passband ripple junk.
- Usable directly in real-time DSP or offline.

All functions return lists of Biquad sections:
    H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal
import numpy as np
from math import pi, sin, cos


# =========================================================
# Basic data structure
# =========================================================

@dataclass
class Biquad:
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


# =========================================================
# Core RBJ biquad designs
# =========================================================

def rbj_peak(f0: float, fs: float, Q: float, gain_db: float) -> Biquad:
    """RBJ peaking EQ."""
    A = 10.0**(gain_db / 40.0)
    w0 = 2.0 * pi * (f0 / fs)
    alpha = sin(w0) / (2.0 * Q)
    cw = cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return Biquad(b0, b1, b2, a1, a2)


def rbj_lowshelf(f0: float, fs: float, gain_db: float, slope: float = 1.0) -> Biquad:
    """RBJ low shelf."""
    A = 10.0**(gain_db / 40.0)
    w0 = 2.0 * pi * (f0 / fs)
    cw = cos(w0)
    sw = sin(w0)
    S = max(slope, 1e-6)

    alpha = sw / 2.0 * ((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)**0.5
    Ap1 = A + 1.0
    Am1 = A - 1.0
    Ap1cw = Ap1 * cw
    Am1cw = Am1 * cw

    b0 =    A * (Ap1 - Am1cw + 2.0 * alpha)**1.0
    b1 =  2*A * (Am1 - Ap1cw)
    b2 =    A * (Ap1 - Am1cw - 2.0 * alpha)
    a0 =        Ap1 + Am1cw + 2.0 * alpha
    a1 =   -2 * (Am1 + Ap1cw)
    a2 =        Ap1 + Am1cw - 2.0 * alpha

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return Biquad(b0, b1, b2, a1, a2)


def rbj_highshelf(f0: float, fs: float, gain_db: float, slope: float = 1.0) -> Biquad:
    """RBJ high shelf."""
    A = 10.0**(gain_db / 40.0)
    w0 = 2.0 * pi * (f0 / fs)
    cw = cos(w0)
    sw = sin(w0)
    S = max(slope, 1e-6)

    alpha = sw / 2.0 * ((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)**0.5
    Ap1 = A + 1.0
    Am1 = A - 1.0
    Ap1cw = Ap1 * cw
    Am1cw = Am1 * cw

    b0 =    A * (Ap1 + Am1cw + 2.0 * alpha)
    b1 = -2*A * (Am1 + Ap1cw)
    b2 =    A * (Ap1 + Am1cw - 2.0 * alpha)
    a0 =        Ap1 - Am1cw + 2.0 * alpha
    a1 =    2 * (Am1 - Ap1cw)
    a2 =        Ap1 - Am1cw - 2.0 * alpha

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return Biquad(b0, b1, b2, a1, a2)


# =========================================================
# Butterworth-style Q distribution
# =========================================================

def butterworth_Qs(order: int) -> List[float]:
    """
    Q values for 2nd-order sections of an Nth-order Butterworth.

    We use them as shaping-Qs for high-order peaking EQ.

    For Butterworth poles on unit circle:
        p_k = -sin(theta_k) + j cos(theta_k),
        theta_k = (2k-1)π/(2N)
      → ζ_k = -Re(p_k) = sin(theta_k)
      → Q_k = 1/(2 ζ_k) = 1/(2 sin(theta_k)).
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("Butterworth order must be even and >= 2.")

    n_sections = order // 2
    Qs: List[float] = []
    for k in range(1, n_sections + 1):
        theta = (2 * k - 1) * pi / (2.0 * order)
        s_theta = sin(theta)
        if s_theta <= 0:
            s_theta = 1e-6
        Q = 1.0 / (2.0 * s_theta)
        Qs.append(Q)
    return Qs


# =========================================================
# High-order peaking EQ (Butterworth-style)
# =========================================================

def design_highorder_peak(
    order: int,
    fs: float,
    f0: float,
    gain_db: float,
) -> List[Biquad]:
    """
    High-order peaking EQ.

    - order: even total order (2,4,6,8,...).
      Each biquad is 2nd order ⇒ num_stages = order / 2.
    - fs: sample rate
    - f0: center frequency
    - gain_db: overall desired boost/cut at f0 (dB)

    Implementation:
      - Take Butterworth Q_k for given 'order'.
      - Build RBJ peaking biquads at same f0.
      - Split gain evenly in dB per stage so that the
        product response at f0 matches 'gain_db'.

    This gives a smooth bell that gets tighter/steeper
    as 'order' increases. No notches, no ripple.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("order must be even and >= 2.")
    if not (0 < f0 < fs * 0.5):
        raise ValueError("f0 must be between 0 and Nyquist.")

    Qs = butterworth_Qs(order)
    n = len(Qs)
    per_stage_gain_db = gain_db / n

    biquads = [
        rbj_peak(f0=f0, fs=fs, Q=Q, gain_db=per_stage_gain_db)
        for Q in Qs
    ]
    return biquads


# =========================================================
# High-order shelves
# =========================================================

def design_highorder_lowshelf(
    order: int,
    fs: float,
    f0: float,
    gain_db: float,
) -> List[Biquad]:
    """
    High-order low shelf using cascaded RBJ low shelves.

    - order: even (2,4,6,...). Stages = order/2.
    - Corner at f0, total gain = gain_db at low frequencies.
    - Each stage uses gain_db / num_stages.

    This yields a steeper but still smooth Butterworth-like shelf.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("order must be even and >= 2.")
    if not (0 < f0 < fs * 0.5):
        raise ValueError("f0 must be between 0 and Nyquist.")

    n = order // 2
    per_stage_gain_db = gain_db / n

    biquads = [
        rbj_lowshelf(f0=f0, fs=fs, gain_db=per_stage_gain_db, slope=1.0)
        for _ in range(n)
    ]
    return biquads


def design_highorder_highshelf(
    order: int,
    fs: float,
    f0: float,
    gain_db: float,
) -> List[Biquad]:
    """
    High-order high shelf using cascaded RBJ high shelves.

    Same pattern as design_highorder_lowshelf.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("order must be even and >= 2.")
    if not (0 < f0 < fs * 0.5):
        raise ValueError("f0 must be between 0 and Nyquist.")

    n = order // 2
    per_stage_gain_db = gain_db / n

    biquads = [
        rbj_highshelf(f0=f0, fs=fs, gain_db=per_stage_gain_db, slope=1.0)
        for _ in range(n)
    ]
    return biquads


# =========================================================
# Band-shelf (mid band) via dual shelves
# =========================================================

def design_highorder_bandshelf(
    order: int,
    fs: float,
    f_lo: float,
    f_hi: float,
    gain_db: float,
) -> List[Biquad]:
    """
    Band-shelf (a broad mid boost/cut) built from two shelves:

        - Low shelf at f_lo with +gain_db
        - High shelf at f_hi with -gain_db

    Positive gain_db → boost band between f_lo and f_hi.
    Negative gain_db → cut band between f_lo and f_hi.

    'order' is total; split evenly between the two shelves.

    This is simple, robust, and musically useful.
    """
    if order < 4 or order % 2 != 0:
        raise ValueError("order must be even and >= 4.")
    if not (0 < f_lo < f_hi < fs * 0.5):
        raise ValueError("Require 0 < f_lo < f_hi < Nyquist.")

    half_order = order // 2
    # half of sections for low shelf, half for high shelf
    low_order = high_order = half_order

    # For a positive gain:
    #   LS +G at low end, HS -G at high end → plateau in middle
    # For negative gain: signs flip but logic same.
    ls = design_highorder_lowshelf(low_order, fs, f_lo, gain_db)
    hs = design_highorder_highshelf(high_order, fs, f_hi, -gain_db)

    return ls + hs


# =========================================================
# Utility: cascade frequency response
# =========================================================

def cascade_freq_response(
    biquads: List[Biquad],
    fs: float,
    n_fft: int = 4096,
):
    """Compute complex frequency response of a cascade of biquads."""
    w = np.linspace(0.0, pi, n_fft)
    z = np.exp(1j * w)
    H = np.ones_like(z, dtype=complex)

    for bq in biquads:
        H *= (bq.b0 + bq.b1 / z + bq.b2 / (z**2)) / (1 + bq.a1 / z + bq.a2 / (z**2))

    f = w * fs / (2.0 * pi)
    return f, H


# =========================================================
# Demo / quick test
# =========================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48000.0
    f0 = 2000.0
    gain_db = 6.0

    # Compare 2nd vs 8th order peaking EQ
    peq2 = design_highorder_peak(2, fs, f0, gain_db)
    peq8 = design_highorder_peak(8, fs, f0, gain_db)

    f, H2 = cascade_freq_response(peq2, fs)
    _, H8 = cascade_freq_response(peq8, fs)

    mag2 = 20 * np.log10(np.maximum(np.abs(H2), 1e-12))
    mag8 = 20 * np.log10(np.maximum(np.abs(H8), 1e-12))

    plt.figure(figsize=(8, 4))
    plt.semilogx(f, mag2, label="Peak order 2 (classic RBJ)")
    plt.semilogx(f, mag8, label="Peak order 8 (Butterworth-style)")
    plt.axvline(f0, color="gray", ls="--", alpha=0.3)
    plt.grid(which="both", alpha=0.3)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"High-Order Peaking EQ (+{gain_db} dB @ {int(f0)} Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Example high-order shelves
    ls = design_highorder_lowshelf(8, fs, 200.0, 6.0)
    hs = design_highorder_highshelf(8, fs, 6000.0, -6.0)
    bs = design_highorder_bandshelf(8, fs, 500.0, 4000.0, 4.0)

    for label, bqs in [
        ("Low-shelf +6dB @200Hz, order8", ls),
        ("High-shelf -6dB @6kHz, order8", hs),
        ("Band-shelf +4dB 500–4kHz, order8", bs),
    ]:
        f, H = cascade_freq_response(bqs, fs)
        mag = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
        plt.figure(figsize=(8, 3))
        plt.semilogx(f, mag, label=label)
        plt.grid(which="both", alpha=0.3)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title(label)
        plt.tight_layout()
        plt.show()

```
