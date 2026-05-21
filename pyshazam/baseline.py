"""BASELINe antigen-driven selection analysis (R/Baseline.R)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from . import core
from .core import CONST_I, BAYESIAN_FITTED, makeNullRegionDefinition
from .mutation import observedMutations
from .expected import expectedMutations
from .targeting import HH_S5F

_MAX_SIGMA = 20
_LEN_SIGMA = 4001


# ---------------------------------------------------------------------------
# Baseline object
# ---------------------------------------------------------------------------
class Baseline:
    """Container for BASELINe selection PDFs and statistics."""

    def __init__(self, description="", db=None, regionDefinition=None,
                 testStatistic="", regions=None, numbOfSeqs=None,
                 binomK=None, binomN=None, binomP=None, pdfs=None,
                 stats=None):
        self.description = description
        self.db = db if db is not None else pd.DataFrame()
        if regionDefinition is None:
            regionDefinition = makeNullRegionDefinition()
        self.regionDefinition = regionDefinition
        self.testStatistic = testStatistic
        self.regions = (list(regions) if regions is not None
                        else list(regionDefinition.regions))
        self.numbOfSeqs = numbOfSeqs
        self.binomK = binomK
        self.binomN = binomN
        self.binomP = binomP
        self.pdfs = pdfs if pdfs is not None else {}
        self.stats = stats if stats is not None else pd.DataFrame()

    def __repr__(self):
        return (f"Baseline(testStatistic={self.testStatistic!r}, "
                f"regions={self.regions})")


def createBaseline(description="", db=None, regionDefinition=None,
                   testStatistic="", regions=None, numbOfSeqs=None,
                   binomK=None, binomN=None, binomP=None, pdfs=None,
                   stats=None):
    """Create a Baseline object."""
    return Baseline(description=description, db=db,
                    regionDefinition=regionDefinition,
                    testStatistic=testStatistic, regions=regions,
                    numbOfSeqs=numbOfSeqs, binomK=binomK, binomN=binomN,
                    binomP=binomP, pdfs=pdfs, stats=stats)


def editBaseline(baseline, field, value):
    """Edit a field of a Baseline object."""
    if not hasattr(baseline, field):
        raise ValueError(f"{field} is not part of the Baseline object.")
    setattr(baseline, field, value)
    return baseline


# ---------------------------------------------------------------------------
# Binomial PDF (the BASELINe posterior)
# ---------------------------------------------------------------------------
def calcBaselineBinomialPdf(x=3, n=10, p=0.33, max_sigma=_MAX_SIGMA,
                            length_sigma=_LEN_SIGMA):
    """BASELINe posterior PDF over selection strength sigma.

    Faithful port of shazam's ``calcBaselineBinomialPdf``.
    """
    if n == 0:
        return None
    sigma_s = np.linspace(-max_sigma, max_sigma, length_sigma)
    const_i = CONST_I
    sigma_1 = np.log((const_i / (1 - const_i)) / (p / (1 - p)))
    index = min(int(n), 60)
    bf = BAYESIAN_FITTED[index - 1]
    y = (beta_dist.pdf(const_i, x + bf, n + bf - x)
         * (1 - p) * p * np.exp(sigma_1)
         / ((1 - p) ** 2 + 2 * p * (1 - p) * np.exp(sigma_1)
            + (p ** 2) * np.exp(2 * sigma_1)))
    if np.any(np.isnan(y)):
        return None
    # linear interpolation onto sigma_s (R approx; sigma_1 ascending? sort)
    order = np.argsort(sigma_1)
    tmp = np.interp(sigma_s, sigma_1[order], y[order],
                    left=np.nan, right=np.nan)
    norm = np.nansum(tmp)
    return tmp / norm / (2 * max_sigma / (length_sigma - 1))


# ---------------------------------------------------------------------------
# calcBaselineHelper
# ---------------------------------------------------------------------------
def _grep(pattern, names):
    import re
    pat = re.compile(pattern)
    return [n for n in names if pat.search(n)]


def calcBaselineHelper(observed, expected, region, testStatistic="local",
                       regionDefinition=None):
    """Compute the BASELINe PDF for a single sequence-region.

    ``observed`` / ``expected`` are dict-like (column name -> value).
    Returns (pdf_array, obsX, obsN, expP).
    """
    if regionDefinition is None:
        regions = makeNullRegionDefinition().regions
    else:
        regions = regionDefinition.regions
    if testStatistic == "focused" and len(regions) != 2:
        testStatistic = "local"

    obs_names = list(observed.keys())
    exp_names = list(expected.keys())

    if testStatistic == "local":
        obsX = _grep(f"mu_count_{region}_r", obs_names)
        obsN = _grep(f"mu_count_{region}_", obs_names)
        expX = _grep(f"mu_expected_{region}_r", exp_names)
        expN = _grep(f"mu_expected_{region}_", exp_names)
    elif testStatistic == "focused":
        other = [r for r in regions if r != region]
        obsX = _grep(f"mu_count_{region}_r", obs_names)
        obsN = _grep(f"mu_count_{region}|"
                     + "|".join(f"mu_count_{o}_s" for o in other), obs_names)
        expX = _grep(f"mu_expected_{region}_r", exp_names)
        expN = _grep(f"mu_expected_{region}|"
                     + "|".join(f"mu_expected_{o}_s" for o in other),
                     exp_names)
    else:  # imbalanced
        obsX = _grep(f"mu_count_{region}", obs_names)
        obsN = _grep("mu_count_", obs_names)
        expX = _grep(f"mu_expected_{region}", exp_names)
        expN = _grep("mu_expected_", exp_names)

    oX = float(np.nansum([observed[c] for c in obsX]))
    oN = float(np.nansum([observed[c] for c in obsN]))
    eX = np.nansum([expected[c] for c in expX])
    eN = np.nansum([expected[c] for c in expN])
    eP = float(eX / eN) if eN else np.nan

    pdf = calcBaselineBinomialPdf(x=oX, n=oN, p=eP)
    return pdf, oX, oN, eP


# ---------------------------------------------------------------------------
# Statistics on a PDF
# ---------------------------------------------------------------------------
def baselineSigma(base, max_sigma=_MAX_SIGMA, length_sigma=_LEN_SIGMA):
    """Mean selection strength sigma of a BASELINe PDF."""
    base = np.asarray(base, dtype=float)
    if np.any(np.isnan(base)):
        return np.nan
    sigma_s = np.linspace(-max_sigma, max_sigma, length_sigma)
    norm = np.nansum(base)
    return float(base @ sigma_s / norm)


def baselineCI(base, low=0.025, up=0.975, max_sigma=_MAX_SIGMA,
               length_sigma=_LEN_SIGMA):
    """Confidence interval bounds of a BASELINe PDF."""
    base = np.asarray(base, dtype=float)
    if np.any(np.isnan(base)):
        return (np.nan, np.nan)
    sigma_s = np.linspace(-max_sigma, max_sigma, length_sigma)
    cdf = np.cumsum(base)
    cdf = cdf / cdf[-1]

    def find_interval(v):
        # R findInterval: largest 1-based i with cdf[i] <= v
        return int(np.sum(cdf <= v))

    # R uses 1-based indexing; convert R index k -> python index k-1
    iLow = find_interval(low)        # 1-based
    iUp = find_interval(up)          # 1-based
    fLow = ((low - cdf[iLow - 1])
            / (cdf[iLow] - cdf[iLow - 1]))
    fUp = ((up - cdf[iUp - 1])
           / (cdf[iUp - 1] - cdf[iUp - 2]))
    sLow = (sigma_s[iLow - 1]
            + fLow * (sigma_s[iLow] - sigma_s[iLow - 1]))
    sUp = (sigma_s[iUp - 1]
           + fUp * (sigma_s[iUp - 1] - sigma_s[iUp - 2]))
    return (float(sLow), float(sUp))


def baselinePValue(base, length_sigma=_LEN_SIGMA, max_sigma=_MAX_SIGMA):
    """Signed p-value that a BASELINe PDF differs from zero selection."""
    base = np.asarray(base, dtype=float)
    if np.any(np.isnan(base)):
        return np.nan
    norm = np.nansum(base)
    half = (length_sigma - 1) // 2
    pvalue = (np.sum(base[:half]) + base[half] / 2) / norm
    if pvalue > 0.5:
        pvalue = -(1 - pvalue)
    return float(pvalue)


def baseline2DistPValue(base1, base2):
    """Two-sided p-value that two BASELINe PDFs differ."""
    base1 = np.asarray(base1, dtype=float)
    base2 = np.asarray(base2, dtype=float)
    if len(base1) != len(base2):
        raise ValueError("base1 and base2 must be the same length.")
    if np.all(np.isnan(base1)) or np.all(np.isnan(base2)):
        return np.nan
    base1 = base1 / np.nansum(base1)
    base2 = base2 / np.nansum(base2)
    cum2 = np.cumsum(base2) - base2 / 2
    pvalue = float(np.sum(base1 * cum2))
    if pvalue > 0.5:
        pvalue = 1 - pvalue
    return pvalue


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------
def PowersOfTwo(G=100):
    """Decompose G into a sum of powers of two (returns exponents)."""
    exps = []
    G = int(G)
    while G > 0:
        e = int(np.floor(np.log2(G)))
        exps.append(e)
        G -= 2 ** e
    return exps


def break2chunks(G=1000):
    """Break G into chunks of base size (port of shazam helper)."""
    base = 2 ** int(round(np.log2(np.sqrt(G))))
    return [base] * (G // base - 1) + [base + G - (G // base) * base]


def _approx(y, n_out):
    """1-D linear resample of vector y onto n_out equally-spaced points."""
    x_in = np.arange(1, len(y) + 1)
    x_out = np.linspace(1, len(y), n_out)
    return np.interp(x_out, x_in, y)


def weighted_conv(x, y, w=1, m=100, length_sigma=_LEN_SIGMA):
    """Weighted convolution of two PDFs (port of shazam's weighted_conv)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lx, ly = len(x), len(y)
    if (lx < m) or (lx * w < m) or (ly < m) or (ly * w < m):
        if w < 1:
            y1 = _approx(y, m)
            x1 = _approx(x, int(round(m / w)))
        else:
            y1 = _approx(y, int(round(m * w)))
            x1 = _approx(x, m)
    else:
        x1 = x
        y1 = _approx(y, int(np.floor(lx * w)))
    lx1, ly1 = len(x1), len(y1)
    conv = np.convolve(x1, y1)
    tmp = _approx_xy(np.arange(1, lx1 + ly1), conv, length_sigma)
    tmp[tmp <= 0] = 0
    s = tmp.sum()
    return tmp / s if s > 0 else tmp


def _approx_xy(x, y, n_out):
    x_out = np.linspace(x[0], x[-1], n_out)
    return np.interp(x_out, x, y)


def convolutionPowersOfTwo(cons, length_sigma=_LEN_SIGMA):
    """Convolve a 2^k-column matrix of PDFs pairwise down to one PDF."""
    cons = np.array(cons, dtype=float)
    G = cons.shape[1]
    if G > 1:
        gen = int(np.log2(G))
        while gen >= 1:
            ll = np.arange(2, 2 ** gen + 1, 2)
            for l in ll:
                cons[:, l // 2 - 1] = weighted_conv(
                    cons[:, l - 1], cons[:, l - 2],
                    length_sigma=length_sigma)
            gen -= 1
    return cons[:, 0]


def convolutionPowersOfTwoByTwos(cons, length_sigma=_LEN_SIGMA, G=1):
    """Group convolution by powers of two."""
    cons = np.array(cons, dtype=float)
    if cons.ndim == 2:
        G = cons.shape[1]
    groups = PowersOfTwo(G)
    nrow = cons.size // G if cons.ndim == 2 else len(cons)
    matG = np.full((nrow, len(groups)), np.nan)
    start = 0   # 0-based; R 'startIndex' is 1-based
    for i, g in enumerate(groups):
        stop = 2 ** g + start   # 0-based exclusive end == R stopIndex
        if stop != start + 1:
            matG[:, i] = convolutionPowersOfTwo(
                cons[:, start:stop], length_sigma=length_sigma)
            start = stop
        else:
            # R: single-column group; startIndex is NOT advanced (bug-faithful)
            if G > 1:
                matG[:, i] = cons[:, start]
            else:
                matG[:, i] = np.ravel(cons)
    return matG, groups


def calculate_bayesGHelper(listMatG, length_sigma=_LEN_SIGMA):
    """Weighted recursive convolution (port of shazam helper)."""
    matG, groups = listMatG
    matG = np.asarray(matG, dtype=float)
    resConv = matG[:, 0]
    denom = 2 ** groups[0]
    for i in range(1, len(groups)):
        resConv = weighted_conv(resConv, matG[:, i],
                                w=(2 ** groups[i]) / denom,
                                length_sigma=length_sigma)
        denom += 2 ** groups[i]
    return resConv


def fastConv(cons, max_sigma=_MAX_SIGMA, length_sigma=_LEN_SIGMA):
    """Fast convolution of many PDFs (port of shazam's fastConv)."""
    cons = np.asarray(cons, dtype=float)
    ncol = cons.shape[1]
    chunks = break2chunks(ncol)
    if ncol == 3:
        chunks = [2, 1]
    idx_end = np.cumsum(chunks)
    idx_start = np.concatenate([[0], idx_end[:-1]])
    case = int(np.sum(np.array(chunks) != chunks[0]))
    if case == 1:
        End = max(1, len(chunks) - 1)
    else:
        End = max(1, len(chunks))
    firsts = []
    for i in range(End):
        sub = cons[:, idx_start[i]:idx_end[i]]
        firsts.append(convolutionPowersOfTwoByTwos(sub)[0])
    firsts = np.column_stack(firsts) if len(firsts) > 1 else \
        np.asarray(firsts[0])
    if case == 0:
        result = calculate_bayesGHelper(
            convolutionPowersOfTwoByTwos(firsts))
    else:
        li = len(chunks) - 1
        last_sub = cons[:, idx_start[li]:idx_end[li]]
        last = calculate_bayesGHelper(
            convolutionPowersOfTwoByTwos(last_sub))
        result_first = calculate_bayesGHelper(
            convolutionPowersOfTwoByTwos(firsts))
        result = calculate_bayesGHelper((
            np.column_stack([result_first, last]),
            [np.log2(idx_end[li - 1]),
             np.log2(idx_end[li] - idx_start[li])]))
    return np.asarray(result)


def groupPosteriors(listPosteriors, max_sigma=_MAX_SIGMA,
                    length_sigma=_LEN_SIGMA, Threshold=2):
    """Convolve a list of equally-weighted PDFs into a single PDF."""
    listPosteriors = [p for p in listPosteriors if p is not None
                      and not np.all(np.isnan(p))]
    L = len(listPosteriors)
    if L == 0:
        return None
    if L == 1:
        return np.asarray(listPosteriors[0])
    cons = np.column_stack(listPosteriors)
    if L <= Threshold:
        matG = convolutionPowersOfTwoByTwos(cons, length_sigma=length_sigma)
        y = calculate_bayesGHelper(matG, length_sigma=length_sigma)
    else:
        y = fastConv(cons, max_sigma=max_sigma, length_sigma=length_sigma)
    return y / np.sum(y) / (2 * max_sigma / (length_sigma - 1))


# ---------------------------------------------------------------------------
# calcBaseline / groupBaseline / summarizeBaseline / testBaseline
# ---------------------------------------------------------------------------
def calcBaseline(db, sequenceColumn="clonal_sequence",
                 germlineColumn="clonal_germline", testStatistic="local",
                 regionDefinition=None, targetingModel=None,
                 mutationDefinition=None, calcStats=False,
                 cloneColumn=None, juncLengthColumn=None):
    """Calculate per-sequence BASELINe selection PDFs.

    Faithful port of shazam's ``calcBaseline``.
    """
    if targetingModel is None:
        targetingModel = HH_S5F
    db = db.copy().reset_index(drop=True)

    if regionDefinition is None:
        rd_labels = makeNullRegionDefinition().labels
        regions = makeNullRegionDefinition().regions
    else:
        rd_labels = regionDefinition.labels
        regions = regionDefinition.regions
    observedColumns = [f"mu_count_{lab}" for lab in rd_labels]
    expectedColumns = [f"mu_expected_{lab}" for lab in rd_labels]

    if not all(c in db.columns for c in observedColumns + expectedColumns):
        db = observedMutations(
            db, sequenceColumn=sequenceColumn, germlineColumn=germlineColumn,
            regionDefinition=regionDefinition,
            mutationDefinition=mutationDefinition, frequency=False,
            combine=False, cloneColumn=cloneColumn,
            juncLengthColumn=juncLengthColumn or "junction_length")
        db = expectedMutations(
            db, sequenceColumn=sequenceColumn, germlineColumn=germlineColumn,
            regionDefinition=regionDefinition, targetingModel=targetingModel,
            mutationDefinition=mutationDefinition, cloneColumn=cloneColumn,
            juncLengthColumn=juncLengthColumn or "junction_length")

    cols_obs = [c for c in db.columns if c.startswith("mu_count_")]
    cols_exp = [c for c in db.columns if c.startswith("mu_expected_")]
    n = len(db)
    rd_name = regionDefinition.name if regionDefinition is not None else ""

    list_pdfs = {r: np.full((n, _LEN_SIGMA), np.nan) for r in regions}
    binomK = {r: np.full(n, np.nan) for r in regions}
    binomN = {r: np.full(n, np.nan) for r in regions}
    binomP = {r: np.full(n, np.nan) for r in regions}
    numbOfSeqs = {r: np.ones(n) for r in regions}

    for region in regions:
        for idx in range(n):
            rd = regionDefinition
            if rd_name in ("IMGT_VDJ_BY_REGIONS", "IMGT_VDJ"):
                from .regions import setRegionBoundaries
                rd = setRegionBoundaries(
                    juncLength=db[juncLengthColumn].iloc[idx],
                    sequenceImgt=db[sequenceColumn].iloc[idx],
                    regionDefinition=regionDefinition)
            observed = {c: db[c].iloc[idx] for c in cols_obs}
            expected = {c: db[c].iloc[idx] for c in cols_exp}
            pdf, k, nn, p = calcBaselineHelper(
                observed, expected, region, testStatistic, rd)
            if pdf is not None:
                list_pdfs[region][idx] = pdf
            binomK[region][idx] = k
            binomN[region][idx] = nn
            binomP[region][idx] = p
        numbOfSeqs[region][np.isnan(binomK[region])] = 0

    return createBaseline(
        description="", db=db, regionDefinition=regionDefinition,
        testStatistic=testStatistic, regions=regions,
        numbOfSeqs=numbOfSeqs, binomK=binomK, binomN=binomN, binomP=binomP,
        pdfs=list_pdfs)


def groupBaseline(baseline, groupBy):
    """Convolve BASELINe PDFs within groups (port of shazam's groupBaseline)."""
    if isinstance(groupBy, str):
        groupBy = [groupBy]
    else:
        groupBy = list(groupBy)
    db = baseline.db
    uniqueGroups = db[groupBy].drop_duplicates().reset_index(drop=True)
    n_groups = len(uniqueGroups)
    regions = baseline.regions

    group_idx = []
    for _, grp in uniqueGroups.iterrows():
        mask = np.ones(len(db), dtype=bool)
        for col in groupBy:
            mask &= (db[col].values == grp[col])
        group_idx.append(np.where(mask)[0])

    new_pdfs = {r: np.full((n_groups, _LEN_SIGMA), np.nan) for r in regions}
    numbOfSeqs = {r: np.zeros(n_groups) for r in regions}

    for region in regions:
        for gi, idx in enumerate(group_idx):
            mat = baseline.pdfs[region][idx]
            rows = [mat[k] for k in range(mat.shape[0])
                    if not np.all(np.isnan(mat[k]))]
            nos = baseline.numbOfSeqs[region][idx]
            nos = nos[nos > 0]
            numbOfNonNA = len(rows)
            if numbOfNonNA == 0:
                continue
            if np.sum(nos) == len(nos):
                conv = groupPosteriors(rows)
            else:
                # weighted combination (sort by num of seqs)
                order = np.argsort(nos)
                sorted_rows = [rows[k] for k in order]
                sorted_nos = list(nos[order])
                while True:
                    from collections import Counter
                    cnt = Counter(sorted_nos)
                    dup = [w for w in sorted(cnt) if cnt[w] > 1]
                    if not dup:
                        break
                    pw = dup[0]
                    indices = [k for k, v in enumerate(sorted_nos)
                               if v == pw]
                    same = [sorted_rows[k] for k in indices]
                    updated = groupPosteriors(same)
                    new_weight = pw * len(indices)
                    sorted_rows = [r for k, r in enumerate(sorted_rows)
                                   if k not in indices]
                    sorted_nos = [v for k, v in enumerate(sorted_nos)
                                  if k not in indices]
                    sorted_rows.append(updated)
                    sorted_nos.append(new_weight)
                    order = np.argsort(sorted_nos)
                    sorted_rows = [sorted_rows[k] for k in order]
                    sorted_nos = list(np.array(sorted_nos)[order])
                # combine remaining weighted posteriors pairwise
                conv = sorted_rows[0]
                cum_w = sorted_nos[0]
                for k in range(1, len(sorted_rows)):
                    matG = (np.column_stack([conv, sorted_rows[k]]),
                            [int(np.log2(cum_w)),
                             int(np.log2(sorted_nos[k]))])
                    conv = calculate_bayesGHelper(matG)
                    cum_w += sorted_nos[k]
                conv = conv / np.sum(conv) / (
                    2 * _MAX_SIGMA / (_LEN_SIGMA - 1))
            new_pdfs[region][gi] = conv
            numbOfSeqs[region][gi] = numbOfNonNA

    return createBaseline(
        description=baseline.description, db=uniqueGroups,
        regionDefinition=baseline.regionDefinition,
        testStatistic=baseline.testStatistic, regions=regions,
        numbOfSeqs=numbOfSeqs, pdfs=new_pdfs)


def summarizeBaseline(baseline, returnType="baseline"):
    """Compute per-PDF BASELINe statistics (sigma, CI, p-value)."""
    db = baseline.db
    regions = baseline.regions
    id_col = None
    for c in ("sequence_id", "SEQUENCE_ID"):
        if c in db.columns:
            id_col = c
            break
    rows = []
    for idx in range(len(db)):
        for region in regions:
            pdf = baseline.pdfs[region][idx]
            ci = baselineCI(pdf)
            row = {}
            if id_col is not None:
                row[id_col] = db[id_col].iloc[idx]
            row.update({
                "region": region,
                "baseline_sigma": baselineSigma(pdf),
                "baseline_ci_lower": ci[0],
                "baseline_ci_upper": ci[1],
                "baseline_ci_pvalue": baselinePValue(pdf),
            })
            rows.append(row)
    stats = pd.DataFrame(rows)
    if returnType == "df":
        return stats
    return editBaseline(baseline, "stats", stats)


def testBaseline(baseline, groupBy):
    """Two-sample significance test of grouped BASELINe PDFs."""
    import itertools
    from statsmodels.stats.multitest import multipletests

    groups = [str(g) for g in baseline.db[groupBy]]
    if len(groups) < 2:
        raise ValueError(f"The {groupBy} column needs at least two groups.")
    pairs = list(itertools.combinations(range(len(groups)), 2))
    test_names = [f"{groups[i]} != {groups[j]}" for i, j in pairs]

    rows = []
    for region in baseline.regions:
        d = baseline.pdfs[region]
        for (i, j), name in zip(pairs, test_names):
            pval = baseline2DistPValue(d[i], d[j])
            rows.append({"region": region, "test": name, "pvalue": pval})
    df = pd.DataFrame(rows)
    valid = df["pvalue"].notna()
    df["fdr"] = np.nan
    if valid.sum() > 0:
        df.loc[valid, "fdr"] = multipletests(
            df.loc[valid, "pvalue"], method="fdr_bh")[1]
    return df
