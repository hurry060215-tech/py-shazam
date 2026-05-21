"""matplotlib plotting for pyshazam results."""
from __future__ import annotations

import numpy as np

from .core import REGION_PALETTE

_MAX_SIGMA = 20
_LEN_SIGMA = 4001


def plotDistThreshold(distances, threshold=None, binwidth=0.02, ax=None,
                      title="Distance to nearest", color="steelblue"):
    """Histogram of distance-to-nearest with the fitted threshold marked."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    bins = np.arange(d.min(), d.max() + binwidth, binwidth)
    ax.hist(d, bins=bins, color=color, edgecolor="white")
    if threshold is not None:
        ax.axvline(threshold, color="firebrick", linestyle="--",
                   label=f"threshold = {threshold:.3f}")
        ax.legend()
    ax.set_xlabel("distance")
    ax.set_ylabel("count")
    ax.set_title(title)
    return ax


def plotDensityThreshold(densityThreshold, ax=None,
                         title="Density threshold"):
    """Plot the smoothed density estimate and the inferred threshold."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    dt = densityThreshold
    d = np.asarray(dt.x, dtype=float)
    d = d[np.isfinite(d)]
    ax.hist(d, bins=40, density=True, color="lightgray", edgecolor="white")
    ax.plot(dt.xdens, dt.ydens, color="steelblue", lw=2, label="density")
    if dt.threshold is not None:
        ax.axvline(dt.threshold, color="firebrick", linestyle="--",
                   label=f"threshold = {dt.threshold:.3f}")
    ax.set_xlabel("distance")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    return ax


def plotGmmThreshold(gmmThreshold, binwidth=0.02, ax=None,
                     title="GMM threshold"):
    """Plot the two-component mixture fit and the inferred threshold."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm, gamma as gamma_dist
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    g = gmmThreshold
    d = np.asarray(g.x, dtype=float)
    bins = np.arange(d.min(), d.max() + binwidth, binwidth)
    ax.hist(d, bins=bins, density=True, color="lightgray", edgecolor="white")
    bits = g.model.split("-")
    x = np.linspace(d.min(), d.max(), 500)

    def curve(fam, p1, p2):
        if fam == "norm":
            return norm.pdf(x, p1, p2)
        return gamma_dist.pdf(x, p1, scale=p2)

    ax.plot(x, g.a1 * curve(bits[0], g.b1, g.c1), color="darkblue", lw=2,
            label="curve 1")
    ax.plot(x, g.a2 * curve(bits[1], g.b2, g.c2), color="darkred", lw=2,
            label="curve 2")
    if g.threshold is not None:
        ax.axvline(g.threshold, color="black", linestyle="--",
                   label=f"threshold = {g.threshold:.3f}")
    ax.set_xlabel("distance")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    return ax


def plotMutability(model, nucleotides=("A", "C", "G", "T"), ax=None,
                   title="Mutability"):
    """Bar plot of 5-mer mutability by central nucleotide."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, axes = plt.subplots(len(nucleotides), 1, figsize=(8, 8),
                               sharex=False)
    else:
        axes = [ax] * len(nucleotides)
    mut = (model.mutability if hasattr(model, "mutability")
           else np.asarray(model))
    names = (model.mut_names if hasattr(model, "mut_names")
             else list(getattr(model, "names", [])))
    for k, nuc in enumerate(nucleotides):
        idx = [i for i, nm in enumerate(names) if nm[2] == nuc]
        vals = [mut[i] for i in idx]
        a = axes[k] if hasattr(axes, "__len__") else axes
        a.bar(range(len(vals)), vals, color=REGION_PALETTE.get("cdr"))
        a.set_title(f"{title}: {nuc}")
        a.set_ylabel("mutability")
    return axes


def plotMutabilityHeatmap(model, ax=None, title="Mutability heatmap"):
    """Heatmap of 5-mer mutability arranged by 3' / 5' flanking bases."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    mut = (model.mutability if hasattr(model, "mutability")
           else np.asarray(model))
    names = (model.mut_names if hasattr(model, "mut_names")
             else list(getattr(model, "names", [])))
    nuc = ["A", "C", "G", "T"]
    fivemers = [nm for nm in names
                if all(c in nuc for c in nm)]
    rates = {nm: mut[names.index(nm)] for nm in fivemers}
    rows = [a + b for a in nuc for b in nuc]    # 5' flanks
    cols = [a + b for a in nuc for b in nuc]    # 3' flanks
    grid = np.full((len(rows), len(cols) * 4), np.nan)
    for ci, center in enumerate(nuc):
        for ri, r in enumerate(rows):
            for cj, c in enumerate(cols):
                mer = r + center + c
                if mer in rates:
                    grid[ri, ci * len(cols) + cj] = rates[mer]
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("central nucleotide / 3' flank")
    ax.set_ylabel("5' flank")
    plt.colorbar(im, ax=ax)
    return ax


def plotBaselineDensity(baseline, idColumn=None, sigmaLimits=(-5, 5),
                        ax=None, title="BASELINe selection",
                        colorValues=None):
    """Plot grouped BASELINe selection probability density functions."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sigma_s = np.linspace(-_MAX_SIGMA, _MAX_SIGMA, _LEN_SIGMA)
    db = baseline.db
    for region in baseline.regions:
        mat = baseline.pdfs[region]
        for i in range(mat.shape[0]):
            pdf = mat[i]
            if np.all(np.isnan(pdf)):
                continue
            label = region
            if idColumn is not None and idColumn in db.columns:
                label = f"{db[idColumn].iloc[i]} ({region})"
            color = None
            if colorValues is not None:
                key = (db[idColumn].iloc[i]
                       if idColumn is not None and idColumn in db.columns
                       else region)
                color = colorValues.get(key)
            ax.plot(sigma_s, pdf, label=label, color=color)
    ax.set_xlim(sigmaLimits)
    ax.set_xlabel("selection strength (sigma)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plotBaselineSummary(baseline, idColumn=None, ax=None,
                        title="BASELINe summary"):
    """Bar plot of mean BASELINe selection strength with CI error bars."""
    import matplotlib.pyplot as plt
    from .baseline import summarizeBaseline
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    if baseline.stats is None or len(baseline.stats) == 0:
        stats = summarizeBaseline(baseline, returnType="df")
    else:
        stats = baseline.stats
    x = np.arange(len(stats))
    sigma = stats["baseline_sigma"].values
    lo = stats["baseline_ci_lower"].values
    hi = stats["baseline_ci_upper"].values
    err = np.vstack([sigma - lo, hi - sigma])
    ax.bar(x, sigma, yerr=err, capsize=4, color="steelblue")
    ax.axhline(0, color="black", lw=0.8)
    labels = stats["region"].astype(str).tolist()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("selection strength (sigma)")
    ax.set_title(title)
    return ax


def plotMutationFrequency(db, columns=None, ax=None,
                          title="Mutation frequency"):
    """Box plot of mutation frequency / count columns from a data frame."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if columns is None:
        columns = [c for c in db.columns
                   if c.startswith(("mu_freq", "mu_count"))]
    data = [db[c].dropna().values for c in columns]
    ax.boxplot(data, labels=columns)
    ax.set_ylabel("value")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return ax
