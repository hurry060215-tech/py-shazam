"""Distance-to-nearest and clonal threshold detection.

Faithful port of shazam's R/DistToNearest.R.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm, gamma as gamma_dist

from . import core
from .kedd import h_ucv
from .targeting import (HH_S5F, MK_RS5NF, calcTargetingDistance, HH_S1F,
                        MK_RS1NF, HKL_S1F)

# DNA / AA distance matrices (alakazam getDNAMatrix gap=0 / getAAMatrix)
_DNA_CHARS = list("ACGTN-.")
_AA_CHARS = list("ACDEFGHIKLMNPQRSTVWXY*-.")


def getDNAMatrix(gap=0):
    """Single-nucleotide Hamming distance matrix (alakazam getDNAMatrix)."""
    n = len(_DNA_CHARS)
    m = np.ones((n, n))
    np.fill_diagonal(m, 0)
    # ambiguous / gap characters -> distance 0
    for i, c in enumerate(_DNA_CHARS):
        if c in "N-.":
            m[i, :] = 0
            m[:, i] = 0
    return pd.DataFrame(m, index=_DNA_CHARS, columns=_DNA_CHARS)


def getAAMatrix():
    """Single-amino-acid Hamming distance matrix (alakazam getAAMatrix)."""
    n = len(_AA_CHARS)
    m = np.ones((n, n))
    np.fill_diagonal(m, 0)
    for i, c in enumerate(_AA_CHARS):
        if c in "X-.":
            m[i, :] = 0
            m[:, i] = 0
    return pd.DataFrame(m, index=_AA_CHARS, columns=_AA_CHARS)


# precomputed 1-mer distance matrices used by distToNearest
HH_S1F_Distance = calcTargetingDistance(HH_S1F)
MK_RS1NF_Distance = calcTargetingDistance(MK_RS1NF)
HH_S5F_Distance = calcTargetingDistance(HH_S5F)
MK_RS5NF_Distance = calcTargetingDistance(MK_RS5NF)

# backwards-compatible matrices
HS1F_Compat = pd.DataFrame(
    [[0.00, 2.08, 1.00, 1.75, 0, 0, 0],
     [2.08, 0.00, 1.75, 1.00, 0, 0, 0],
     [1.00, 1.75, 0.00, 2.08, 0, 0, 0],
     [1.75, 1.00, 2.08, 0.00, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0]],
    index=list("ACGTN.-"), columns=list("ACGTN.-"))
M1N_Compat = pd.DataFrame(
    [[0.00, 2.86, 1.00, 2.14, 0, 0, 0],
     [2.86, 0.00, 2.14, 1.00, 0, 0, 0],
     [1.00, 2.14, 0.00, 2.86, 0, 0, 0],
     [2.14, 1.00, 2.86, 0.00, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0]],
    index=list("ACGTN.-"), columns=list("ACGTN.-"))


# ---------------------------------------------------------------------------
# Threshold result objects
# ---------------------------------------------------------------------------
class DensityThreshold:
    """Result of ``findThreshold(method='density')``."""

    def __init__(self, x, bandwidth, xdens, ydens, threshold):
        self.x = np.asarray(x)
        self.bandwidth = bandwidth
        self.xdens = np.asarray(xdens)
        self.ydens = np.asarray(ydens)
        self.threshold = threshold

    def __repr__(self):
        return f"DensityThreshold(threshold={self.threshold})"


class GmmThreshold:
    """Result of ``findThreshold(method='gmm')``."""

    def __init__(self, x, model, cutoff, a1, b1, c1, a2, b2, c2, loglk,
                 threshold, sensitivity, specificity, pvalue):
        self.x = np.asarray(x)
        self.model = model
        self.cutoff = cutoff
        self.a1, self.b1, self.c1 = a1, b1, c1
        self.a2, self.b2, self.c2 = a2, b2, c2
        self.loglk = loglk
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.pvalue = pvalue

    def __repr__(self):
        return f"GmmThreshold(threshold={self.threshold})"


# ---------------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------------
def _translate_dna(seq):
    """Translate a DNA sequence to amino acids (codon table)."""
    aa = []
    for i in range(0, len(seq) - len(seq) % 3, 3):
        codon = seq[i:i + 3]
        a = core.AMINO_ACIDS.get(codon)
        if a is None:
            if "-" in codon or "." in codon:
                a = "-"
            else:
                a = "X"
        aa.append(a)
    return "".join(aa)


def pairwiseDist(sequences, dist_mat):
    """Pairwise distances between equal-length sequences using a char matrix."""
    chars = list(dist_mat.index)
    char_idx = {c: i for i, c in enumerate(chars)}
    M = dist_mat.values
    n = len(sequences)
    arr = np.array([[char_idx.get(c, char_idx.get("N", 0)) for c in s]
                    for s in sequences])
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = M[arr[i], arr[j]].sum()
            out[i, j] = d
            out[j, i] = d
    return out


def _window5mers(seq):
    return [seq[i:i + 5] for i in range(len(seq) - 4)]


def dist5Mers(seq1_5mers, seq2_5mers, targetingDistance, symmetry="avg"):
    """Distance between two sequences broken into 5-mers (port of shazam)."""
    td = targetingDistance
    rows = list(td.index)
    cols = list(td.columns)
    row_idx = {c: i for i, c in enumerate(rows)}
    col_idx = {c: j for j, c in enumerate(cols)}
    M = td.values
    s1, s2 = [], []
    for a, b in zip(seq1_5mers, seq2_5mers):
        if a[2] != b[2]:
            s1.append(a)
            s2.append(b)
    if not s1:
        return 0.0
    d12 = []
    d21 = []
    for a, b in zip(s1, s2):
        # seq1 -> seq2: M[center of seq2, fivemer of seq1]
        d12.append(M[row_idx[b[2]], col_idx[a]])
        d21.append(M[row_idx[a[2]], col_idx[b]])
    d12 = np.array(d12)
    d21 = np.array(d21)
    if symmetry == "avg":
        return float(np.sum((d12 + d21) / 2))
    if symmetry == "min":
        return float(np.sum(np.minimum(d12, d21)))
    return list(d12) + list(d21)


def pairwise5MerDist(sequences, targetingDistance, symmetry="avg"):
    """Pairwise 5-mer distances between sequences."""
    seqs = [re.sub(r"[-.]", "N", s.upper()) for s in sequences]
    seqs = ["NN" + s + "NN" for s in seqs]
    fivemers = [_window5mers(s) for s in seqs]
    n = len(seqs)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dist5Mers(fivemers[i], fivemers[j], targetingDistance,
                          symmetry)
            out[i, j] = d
            out[j, i] = d
    return out


def nearestDist(sequences, model="ham", normalize="none", symmetry="avg",
                crossGroups=None):
    """Distance to closest sequence for each input sequence."""
    sequences = list(sequences)
    if crossGroups is not None and len(set(crossGroups)) < 2:
        return np.full(len(sequences), np.nan)

    seq_uniq = []
    for s in sequences:
        if s not in seq_uniq:
            seq_uniq.append(s)
    n_uniq = len(seq_uniq)
    seq_dist = np.full(len(sequences), np.nan)
    if n_uniq <= 1:
        return seq_dist

    lengths = set(len(s) for s in seq_uniq)
    if len(lengths) > 1:
        raise ValueError("Different sequence lengths found.")
    seq_length = lengths.pop()

    if model == "ham":
        dist_mat = pairwiseDist(seq_uniq, getDNAMatrix(gap=0))
    elif model == "aa":
        aa = [_translate_dna(s) for s in seq_uniq]
        dist_mat = pairwiseDist(aa, getAAMatrix())
    elif model == "hh_s1f":
        dist_mat = pairwiseDist(seq_uniq, HH_S1F_Distance)
    elif model == "mk_rs1nf":
        dist_mat = pairwiseDist(seq_uniq, MK_RS1NF_Distance)
    elif model == "hs1f_compat":
        dist_mat = pairwiseDist(seq_uniq, HS1F_Compat)
    elif model == "m1n_compat":
        dist_mat = pairwiseDist(seq_uniq, M1N_Compat)
    elif model == "hh_s5f":
        dist_mat = pairwise5MerDist(seq_uniq, HH_S5F_Distance, symmetry)
    elif model == "mk_rs5nf":
        dist_mat = pairwise5MerDist(seq_uniq, MK_RS5NF_Distance, symmetry)
    else:
        raise ValueError(f"Unknown model: {model}")

    if normalize == "len":
        dist_mat = dist_mat / seq_length
    elif normalize == "mut":
        raise ValueError('normalize="mut" is not available.')

    uniq_idx = {s: i for i, s in enumerate(seq_uniq)}
    if crossGroups is None:
        seq_uniq_dist = np.full(n_uniq, np.nan)
        for i in range(n_uniq):
            col = dist_mat[:, i]
            gt0 = col[col > 0]
            if len(gt0):
                seq_uniq_dist[i] = gt0.min()
        for k, s in enumerate(sequences):
            seq_dist[k] = seq_uniq_dist[uniq_idx[s]]
    else:
        cg = np.asarray(crossGroups)
        for k, s in enumerate(sequences):
            this_group = cg[k]
            other_seqs = []
            for j, s2 in enumerate(sequences):
                if cg[j] != this_group and s2 not in other_seqs:
                    other_seqs.append(s2)
            this_idx = uniq_idx[s]
            other_idx = [uniq_idx[o] for o in other_seqs]
            if not other_idx:
                continue
            r = dist_mat[other_idx, this_idx]
            gt0 = r[r > 0]
            seq_dist[k] = gt0.min() if len(gt0) else np.nan
    return np.round(seq_dist, 4)


# ---------------------------------------------------------------------------
# V/J/length grouping (port of alakazam::groupGenes core behaviour)
# ---------------------------------------------------------------------------
def _gene_set(call, first):
    """Return the set of gene names from an (ambiguous) allele call."""
    out = []
    for c in str(call).split(","):
        c = c.strip()
        m = re.match(r"([^*]+)", c)
        g = m.group(1) if m else c
        out.append(g)
        if first:
            break
    return set(out)


def _group_genes(db, v_call, j_call, junc_len, first):
    """Assign a vj(l)-group id to each row via union-find over shared calls."""
    n = len(db)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    v_sets = [_gene_set(v, first) for v in db[v_call]]
    j_sets = [_gene_set(j, first) for j in db[j_call]]
    lens = (db[junc_len].tolist() if junc_len is not None
            else [None] * n)
    # bucket by junction length first to limit comparisons
    by_len = {}
    for i in range(n):
        by_len.setdefault(lens[i], []).append(i)
    for L, idxs in by_len.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if (v_sets[i] & v_sets[j]) and (j_sets[i] & j_sets[j]):
                    union(i, j)
    roots = [find(i) for i in range(n)]
    uniq = {}
    groups = []
    for r in roots:
        if r not in uniq:
            uniq[r] = len(uniq) + 1
        groups.append(uniq[r])
    return groups


def distToNearest(db, sequenceColumn="junction", vCallColumn="v_call",
                  jCallColumn="j_call", model="ham", normalize="len",
                  symmetry="avg", first=True, VJthenLen=True, fields=None,
                  cross=None, locusColumn="locus", locusValues=("IGH",),
                  keepVJLgroup=True):
    """Distance to nearest neighbour within V-J-length partitions.

    Faithful port of shazam's ``distToNearest`` (non-single-cell mode).
    """
    db = db.copy().reset_index(drop=True)
    for col in [sequenceColumn, vCallColumn, jCallColumn]:
        db[col] = db[col].astype(str)
    db[sequenceColumn] = db[sequenceColumn].str.upper()
    db["_juncLen"] = db[sequenceColumn].str.len()

    locusValues = [lv.upper() for lv in locusValues]
    n = len(db)
    dist = np.full(n, np.nan)

    # field grouping
    if fields:
        field_grp = db.groupby(list(fields)).ngroup().tolist()
    else:
        field_grp = [0] * n

    # V/J grouping (then-length or simultaneous)
    junc_for_group = None if VJthenLen else "_juncLen"
    vj_group = [None] * n
    for fg in set(field_grp):
        idx = [i for i in range(n) if field_grp[i] == fg]
        sub = db.iloc[idx]
        grps = _group_genes(sub, vCallColumn, jCallColumn, junc_for_group,
                            first)
        for k, i in enumerate(idx):
            vj_group[i] = f"F{fg}_{grps[k]}"
    db["_vj_group"] = vj_group

    # build full groups
    if VJthenLen:
        group_cols = ["_vj_group", "_juncLen", locusColumn]
    else:
        group_cols = ["_vj_group", locusColumn]
    if fields:
        group_cols = group_cols + list(fields)

    for keys, sub in db.groupby(group_cols, sort=False):
        idx = sub.index.tolist()
        loc_vals = db.loc[idx, locusColumn].astype(str).str.upper()
        use = loc_vals.isin(locusValues).values
        sub_idx = [i for i, u in zip(idx, use) if u]
        if len(sub_idx) < 2:
            continue
        seqs = db.loc[sub_idx, sequenceColumn].tolist()
        cg = None
        if cross:
            cg = db.loc[sub_idx].groupby(list(cross)).ngroup().tolist()
        d = nearestDist(seqs, model=model, normalize=normalize,
                        symmetry=symmetry, crossGroups=cg)
        for i, v in zip(sub_idx, d):
            dist[i] = v

    out = db.drop(columns=["_juncLen"]).copy()
    if cross:
        out["cross_dist_nearest"] = dist
    else:
        out["dist_nearest"] = dist
    if (not VJthenLen) and keepVJLgroup:
        out["vjl_group"] = out["_vj_group"]
    out = out.drop(columns=["_vj_group"])
    return out


# ---------------------------------------------------------------------------
# Density-method KDE (port of KernSmooth::bkde, canonical=TRUE, gaussian)
# ---------------------------------------------------------------------------
def _linbin(x, gpoints, truncate=True):
    """Linear binning (port of KernSmooth::linbin)."""
    n = len(x)
    M = len(gpoints)
    a = gpoints[0]
    b = gpoints[-1]
    gcounts = np.zeros(M)
    delta = (b - a) / (M - 1)
    for i in range(n):
        lxi = (x[i] - a) / delta + 1.0    # R 1-based
        li = int(np.floor(lxi))
        rem = lxi - li
        if 1 <= li < M:
            gcounts[li - 1] += 1 - rem
            gcounts[li] += rem
        elif li < 1 and not truncate:
            gcounts[0] += 1
        elif li >= M and not truncate:
            gcounts[M - 1] += 1
    return gcounts


def _bkde(x, bandwidth, gridsize=401, canonical=True):
    """Binned kernel density estimate (faithful port of KernSmooth::bkde).

    Normal kernel only; ``canonical=True`` applies ``h = del0 * bandwidth``.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    M = gridsize
    del0 = (1.0 / (4 * np.pi)) ** (1.0 / 10)
    h = del0 * bandwidth if canonical else bandwidth
    tau = 4.0
    a = x.min() - tau * h
    b = x.max() + tau * h
    gpoints = np.linspace(a, b, M)
    gcounts = _linbin(x, gpoints, truncate=True)
    delta = (b - a) / (h * (M - 1))
    L = min(int(np.floor(tau / delta)), M)
    lvec = np.arange(0, L + 1)
    kappa = norm.pdf(lvec * delta) / (n * h)
    P = 2 ** int(np.ceil(np.log2(M + L + 1)))
    kappa_full = np.concatenate([kappa, np.zeros(P - 2 * L - 1),
                                 kappa[1:][::-1]])
    tot = np.sum(kappa_full) * (b - a) / (M - 1) * n
    gc = np.concatenate([gcounts, np.zeros(P - M)])
    kf = np.fft.fft(kappa_full / tot)
    gcf = np.fft.fft(gc)
    est = (np.real(np.fft.ifft(kf * gcf)))[:M]
    return gpoints, est


def smoothValley(distances):
    """Density-method threshold: minimum of the valley between two modes."""
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    unique_d = np.unique(d)
    if len(unique_d) < 3:
        raise ValueError("smoothValley requires at least 3 unique distances.")
    bandwidth = h_ucv(unique_d, deriv_order=4)["h"]
    xdens, ydens = _bkde(d, bandwidth=bandwidth, canonical=True)
    # find first local minimum
    diff_sign = np.sign(np.diff(ydens))
    second = np.diff(diff_sign)
    valley = np.where(second == 2)[0]
    threshold = xdens[valley[0] + 1] if len(valley) else None
    return DensityThreshold(x=d, bandwidth=bandwidth, xdens=xdens,
                            ydens=ydens, threshold=threshold)


# ---------------------------------------------------------------------------
# GMM-method threshold
# ---------------------------------------------------------------------------
def _gmm_em(ent, edge=0.9):
    """Fit a 2-component gaussian mixture by scanning the valley location."""
    ent = np.asarray(ent, dtype=float)
    ent = ent[np.isfinite(ent)]
    cut = edge * len(ent)
    scan_step = 0.1 if ent.max() <= 5 else 1.0
    nEve = len(ent)

    best = None
    best_lk = 0.0
    valley_loc = 0.0
    while True:
        valley_loc += scan_step
        if np.sum(ent <= valley_loc) > cut:
            break
        low = ent[ent <= valley_loc]
        high = ent[ent > valley_loc]
        if len(low) < 2 or len(high) < 2:
            continue
        omega = np.array([0.5, 0.5])
        mu = np.array([low.mean(), high.mean()])
        sigma = np.array([np.std(ent[ent < valley_loc], ddof=1),
                          np.std(high, ddof=1)])
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            continue
        temp_lk = 0.0
        for _ in range(500):
            # E-step
            resp = np.zeros((nEve, 2))
            for j in range(2):
                resp[:, j] = omega[j] * norm.pdf(ent, mu[j], sigma[j])
            rs = resp.sum(axis=1)
            rs[rs == 0] = 1e-300
            resp = resp / rs[:, None]
            # M-step
            for j in range(2):
                mc = resp[:, j].sum()
                omega[j] = mc / nEve
                mu[j] = np.sum(resp[:, j] * ent) / mc
                sigma[j] = np.sqrt(
                    np.sum(resp[:, j] * (ent - mu[j]) ** 2) / mc)
            log_lk = np.sum(np.log(
                np.maximum(sum(omega[j] * norm.pdf(ent, mu[j], sigma[j])
                               for j in range(2)), 1e-300)))
            err = abs(log_lk - temp_lk)
            if not np.isfinite(err):
                break
            if err < 1e-7:
                break
            temp_lk = log_lk
        if np.isfinite(log_lk) and abs(log_lk) > abs(best_lk):
            best_lk = log_lk
            best = (omega.copy(), mu.copy(), sigma.copy())
    return best


def _norm_area(t1, t2, omega, mu, sigma):
    return omega * (norm.cdf(t2, mu, sigma) - norm.cdf(t1, mu, sigma))


def _gamma_area(t1, t2, omega, k, theta):
    return omega * (gamma_dist.cdf(t2, k, scale=theta)
                    - gamma_dist.cdf(t1, k, scale=theta))


def _curve_pdf(x, fam, p1, p2):
    if fam == "norm":
        return norm.pdf(x, p1, p2)
    return gamma_dist.pdf(x, p1, scale=p2)


def gmmFit(ent, edge=0.9, model="gamma-gamma", cutoff="optimal",
           sen=None, spc=None):
    """Fit a 2-component (gaussian/gamma) mixture and pick a threshold.

    Faithful port of shazam's ``gmmFit`` / ``rocSpace``.
    """
    ent = np.asarray(ent, dtype=float)
    ent = ent[np.isfinite(ent)]
    em = _gmm_em(ent, edge=edge)
    if em is None:
        return None
    omega, mu, sigma = em
    bits = model.split("-")

    # convert moments to curve parameters
    if bits[0] == "norm":
        f1_1, f1_2 = mu[0], sigma[0]
    else:
        f1_1 = (mu[0] / sigma[0]) ** 2
        f1_2 = sigma[0] ** 2 / mu[0]
    if bits[1] == "norm":
        f2_1, f2_2 = mu[1], sigma[1]
    else:
        f2_1 = (mu[1] / sigma[1]) ** 2
        f2_2 = sigma[1] ** 2 / mu[1]
    f1_0 = round(omega[0], 3)

    # MLE refinement of the 5 mixture parameters
    def neg_loglik(params):
        w, p11, p12, p21, p22 = params
        if not (0.001 <= w <= 0.999) or min(p11, p12, p21, p22) <= 0:
            return 1e18
        d = (w * _curve_pdf(ent, bits[0], p11, p12)
             + (1 - w) * _curve_pdf(ent, bits[1], p21, p22))
        d = np.maximum(d, 1e-300)
        return -np.sum(np.log(d))

    from scipy.optimize import minimize
    x0 = [f1_0, f1_1, f1_2, f2_1, f2_2]
    bounds = [(0.001, 0.999), (1e-3, None), (1e-3, None),
              (1e-3, None), (1e-3, None)]
    res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)
    w, p11, p12, p21, p22 = res.x
    loglk = round(abs(-res.fun), 2)

    func1_0, func1_1, func1_2 = w, p11, p12
    func2_0, func2_1, func2_2 = 1 - w, p21, p22

    # order curves: curve1 = lower
    if bits[0] == "norm" and bits[1] == "norm" and func1_1 > func2_1:
        func1_0, func2_0 = func2_0, func1_0
        func1_1, func2_1 = func2_1, func1_1
        func1_2, func2_2 = func2_2, func1_2
    elif (bits[0] == "gamma" and bits[1] == "gamma"
          and func1_1 * func1_2 > func2_1 * func2_2):
        func1_0, func2_0 = func2_0, func1_0
        func1_1, func2_1 = func2_1, func1_1
        func1_2, func2_2 = func2_2, func1_2

    t1, t2 = ent.min(), ent.max()
    minInt = func1_1 if bits[0] == "norm" else func1_1 * func1_2
    maxInt = func2_1 if bits[1] == "norm" else func2_1 * func2_2

    def sens(thr):
        if bits[0] == "norm":
            TP = _norm_area(t1, thr, func1_0, func1_1, func1_2)
            FN = _norm_area(thr, t2, func1_0, func1_1, func1_2)
        else:
            TP = _gamma_area(t1, thr, func1_0, func1_1, func1_2)
            FN = _gamma_area(thr, t2, func1_0, func1_1, func1_2)
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def spec(thr):
        if bits[1] == "norm":
            TN = _norm_area(thr, t2, func2_0, func2_1, func2_2)
            FP = _norm_area(t1, thr, func2_0, func2_1, func2_2)
        else:
            TN = _gamma_area(thr, t2, func2_0, func2_1, func2_2)
            FP = _gamma_area(t1, thr, func2_0, func2_1, func2_2)
        return TN / (TN + FP) if (TN + FP) > 0 else 0.0

    if cutoff == "optimal":
        opt = minimize_scalar(
            lambda thr: -(sens(thr) + spec(thr)) / 2,
            bounds=(minInt, maxInt), method="bounded",
            options={"xatol": 1e-8})
        threshold = opt.x
    elif cutoff == "intersect":
        def diff(thr):
            return (func1_0 * _curve_pdf(thr, bits[0], func1_1, func1_2)
                    - func2_0 * _curve_pdf(thr, bits[1], func2_1, func2_2))
        try:
            threshold = brentq(diff, minInt, maxInt)
        except ValueError:
            lo, hi = minInt * 0.1, maxInt * 5
            threshold = brentq(diff, lo, hi)
    elif cutoff == "user":
        target = sen if sen is not None else spc
        if sen is not None:
            def f(thr):
                return sens(thr) - sen
        else:
            def f(thr):
                return spec(thr) - spc
        threshold = brentq(f, t1, t2)
    else:
        raise ValueError(f"Unknown cutoff: {cutoff}")

    # dip-test p-value (Hartigan)
    try:
        import diptest
        pvalue = diptest.diptest(ent)[1]
    except Exception:
        pvalue = np.nan

    return GmmThreshold(
        x=ent, model=model, cutoff=cutoff,
        a1=func1_0, b1=func1_1, c1=func1_2,
        a2=func2_0, b2=func2_1, c2=func2_2,
        loglk=loglk, threshold=threshold,
        sensitivity=sens(threshold), specificity=spec(threshold),
        pvalue=pvalue)


def findThreshold(distances, method="density", edge=0.9, cross=None,
                  subsample=None, model="gamma-gamma", cutoff="optimal",
                  sen=None, spc=None):
    """Automatically determine the clonal-assignment distance threshold.

    Faithful port of shazam's ``findThreshold``.
    """
    distances = np.asarray(distances, dtype=float)
    if subsample is not None:
        subsample = min(len(distances), subsample)
        distances = np.random.choice(distances, subsample, replace=False)
    if method == "gmm":
        return gmmFit(distances, edge=edge, model=model, cutoff=cutoff,
                      sen=sen, spc=spc)
    if method == "density":
        return smoothValley(distances)
    raise ValueError(f"Unknown method: {method}")
