"""Time-averaged mean squared displacement of one trajectory, by FFT."""

import numpy as np


def tamsd_sums(pos, lags):
    """For each lag D (in samples, D < len(pos)): the sum over origins i of |r(i + D) - r(i)|^2
    and the number of origins, len(pos) - D. By FFT, with
    sum_i |r(i+D) - r(i)|^2 = sum_{i<T-D} |r(i)|^2 + sum_{i>=D} |r(i)|^2 - 2 sum_i r(i).r(i+D)."""
    t = len(pos)
    x = pos - pos.mean(axis=0)
    n = 1 << (2 * t - 1).bit_length()
    f = np.fft.rfft(x, n=n, axis=0)
    auto = np.fft.irfft(np.abs(f) ** 2, n=n, axis=0)[:t].sum(axis=1)
    cum = np.r_[0.0, np.cumsum((x**2).sum(axis=1))]
    d = np.asarray(lags)
    return cum[t - d] + cum[t] - cum[d] - 2 * auto[d], t - d


def tamsd(pos, lags):
    """Mean over origins of |r(i + D) - r(i)|^2 for each lag D (in samples, D < len(pos))."""
    sums, origins = tamsd_sums(pos, lags)
    return sums / origins
