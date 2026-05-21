"""Kernel bandwidth selection (port of shazam's R/kedd.R, gaussian kernel only).

Implements unbiased cross-validation bandwidth selection ``h.ucv`` used by
the density method of ``findThreshold``.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm

_SQRT2 = np.sqrt(2.0)
_SQRTPI = np.sqrt(np.pi)


def _kernel_fun_der(u, deriv_order=0):
    """r-th derivative of the gaussian kernel."""
    r = deriv_order
    u = np.asarray(u, dtype=float)
    if r == 0:
        return norm.pdf(u)
    e = np.exp(-0.5 * u * u)
    if r == 1:
        return -(1 / 2) * u * e * _SQRT2 / _SQRTPI
    if r == 2:
        return (1 / 2) * e * _SQRT2 * (u**2 - 1) / _SQRTPI
    if r == 3:
        return -(1 / 2) * u * e * _SQRT2 * (u**2 - 3) / _SQRTPI
    if r == 4:
        return (1 / 2) * e * _SQRT2 * (u**4 - 6 * u**2 + 3) / _SQRTPI
    if r == 5:
        return -(1 / 2) * u * e * _SQRT2 * (u**4 - 10 * u**2 + 15) / _SQRTPI
    if r == 6:
        return ((1 / 2) * e * _SQRT2
                * (u**6 - 15 * u**4 + 45 * u**2 - 15) / _SQRTPI)
    if r == 7:
        return (-(1 / 2) * u * e * _SQRT2
                * (u**6 - 21 * u**4 + 105 * u**2 - 105) / _SQRTPI)
    if r == 8:
        return ((1 / 2) * e * _SQRT2
                * (u**8 - 28 * u**6 + 210 * u**4 - 420 * u**2 + 105) / _SQRTPI)
    raise NotImplementedError("deriv.order > 8 not supported")


def _kernel_fun_conv(u, deriv_order=0):
    """r-th derivative of the gaussian-kernel self-convolution."""
    r = deriv_order
    u = np.asarray(u, dtype=float)
    if r == 0:
        return norm.pdf(u, loc=0, scale=_SQRT2)
    e = np.exp(-0.25 * u * u)
    if r == 1:
        return (1 / 8) * e * (u**2 - 2) / _SQRTPI
    if r == 2:
        return (1 / 32) * e * (12 - 12 * u**2 + u**4) / _SQRTPI
    if r == 3:
        return (1 / 128) * e * (u**6 - 30 * u**4 + 180 * u**2 - 120) / _SQRTPI
    if r == 4:
        return ((1 / 512) * e
                * (u**8 - 56 * u**6 + 840 * u**4 - 3360 * u**2 + 1680)
                / _SQRTPI)
    if r == 5:
        return ((1 / 2048) * e
                * (u**10 - 90 * u**8 + 2520 * u**6 - 25200 * u**4
                   + 75600 * u**2 - 30240) / _SQRTPI)
    if r == 6:
        return ((1 / 8192) * e
                * (u**12 - 132 * u**10 + 5940 * u**8 - 110880 * u**6
                   + 831600 * u**4 - 1995840 * u**2 + 665280) / _SQRTPI)
    if r == 7:
        return ((1 / 32768) * e
                * (u**14 - 182 * u**12 + 12012 * u**10 - 360360 * u**8
                   + 5045040 * u**6 - 30270240 * u**4 + 60540480 * u**2
                   - 17297280) / _SQRTPI)
    if r == 8:
        return ((1 / 131072) * e
                * (u**16 - 240 * u**14 + 21840 * u**12 - 960960 * u**10
                   + 21621600 * u**8 - 242161920 * u**6 + 1210809600 * u**4
                   - 2075673600 * u**2 + 518918400) / _SQRTPI)
    raise NotImplementedError("deriv.order > 8 not supported")


def _A3_kMr(r):
    val, _ = integrate.quad(
        lambda x: _kernel_fun_der(x, r)**2, -np.inf, np.inf)
    return val


def h_ucv(x, deriv_order=0, lower=None, upper=None, tol=None):
    """Unbiased cross-validation bandwidth (gaussian kernel).

    Faithful port of shazam's ``h.ucv`` for the gaussian kernel.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = np.sort(x)
    n = len(x)
    if n < 3:
        raise ValueError("argument 'x' must have at least 3 data points")
    r = deriv_order
    A2 = 1.0
    A3 = _A3_kMr(r)
    sd = np.std(x, ddof=1)
    hos = ((243 * (2 * r + 1) * A3) / (35 * A2**2)) ** (1.0 / (2 * r + 5)) \
        * sd * n ** (-1.0 / (2 * r + 5))
    if lower is None:
        lower = 0.1 * hos
    if upper is None:
        upper = 2 * hos
    if tol is None:
        tol = 0.1 * lower

    R_Kr1 = A3
    diff = np.subtract.outer(x, x)

    def fucv(h):
        D = _kernel_fun_der(diff / h, deriv_order=2 * r)
        np.fill_diagonal(D, 0.0)
        D = ((-1) ** r / ((n - 1) * h ** (2 * r + 1))) * D.sum(axis=0)
        D1 = np.mean(D)
        D2 = _kernel_fun_conv(diff / h, deriv_order=r)
        np.fill_diagonal(D2, 0.0)
        D3 = ((-1) ** r / ((n - 1) * h ** (2 * r + 1))) * D2.sum(axis=0)
        D4 = np.mean(D3)
        return (1.0 / (n * h ** (2 * r + 1))) * R_Kr1 + D4 - 2 * D1

    res = optimize.minimize_scalar(
        fucv, bounds=(lower, upper), method="bounded",
        options={"xatol": tol})
    return {"h": res.x, "min_ucv": res.fun, "n": n,
            "kernel": "gaussian", "deriv_order": r}
