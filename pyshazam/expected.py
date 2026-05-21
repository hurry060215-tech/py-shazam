"""Expected mutation frequencies (R/MutationProfiling.R: expectedMutations)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import core
from ._loader import codon_table
from .core import NUCLEOTIDES, makeNullRegionDefinition
from .targeting import HH_S5F, TargetingModel, CODON_TABLE_ARR, CODON_TABLE_COLS

_NUC5 = NUCLEOTIDES[:5]
_CT_COL_IDX = {c: j for j, c in enumerate(CODON_TABLE_COLS)}


def calculateTargeting(germlineSeq, inputSeq=None, targetingModel=None,
                       regionDefinition=None):
    """Per-position targeting probabilities for a germline sequence.

    Faithful port of shazam's ``calculateTargeting``. Returns a 5xN DataFrame
    (rows A,C,G,T,N) indexed by germline nucleotides.
    """
    if targetingModel is None:
        targetingModel = HH_S5F

    if inputSeq is not None:
        len_in = len(inputSeq)
        len_gl = len(germlineSeq)
        if regionDefinition is not None:
            L = regionDefinition.seqLength
        else:
            L = max(len_in, len_gl)
        short = min(len_in, len_gl, L)
        c_in = list(inputSeq)[:short]
        c_gl = list(germlineSeq)[:short]
        if short < L:
            fill = ["N"] * (L - short)
            c_in += fill
            c_gl += fill
        for i, ch in enumerate(c_in):
            if ch == "N" or ch not in (_NUC5 + ["."]):
                c_gl[i] = "N"
        s_gl = "".join(c_gl)
    else:
        s_gl = germlineSeq

    # clean IMGT gaps
    gapless = s_gl.replace("...", "ZZZ").replace(".", "N").replace("ZZZ", "...")
    c_gl = list(gapless)
    targeting = np.full((5, len(gapless)), np.nan)

    # remove triplet gaps to make 5-mers
    gapless_nodots = gapless.replace("...", "")
    padded = "NN" + gapless_nodots + "NN"
    n = len(padded)
    tar_cols = list(targetingModel.tar_colnames)
    tar_col_idx = {c: j for j, c in enumerate(tar_cols)}
    tar = targetingModel.targeting
    gapless_targeting = np.full((5, len(gapless_nodots)), np.nan)
    for k in range(len(gapless_nodots)):
        sub = padded[k:k + 5]
        j = tar_col_idx.get(sub)
        if j is not None:
            gapless_targeting[:, k] = tar[:, j]

    non_dot = [i for i, ch in enumerate(c_gl) if ch != "."]
    for col, i in enumerate(non_dot):
        targeting[:, i] = gapless_targeting[:, col]

    targeting[~np.isfinite(targeting)] = np.nan
    return pd.DataFrame(targeting, index=_NUC5, columns=c_gl)


def calculateMutationalPaths(germlineSeq, inputSeq=None,
                             regionDefinition=None, codonTable=None):
    """Per-codon R/S mutation classification matrix (4 x len)."""
    ct_arr, ct_rn, ct_cn = codon_table()
    if codonTable is not None:
        ct_arr = codonTable
        ct_cn = CODON_TABLE_COLS
    col_idx = {c: j for j, c in enumerate(ct_cn)}

    if inputSeq is not None:
        len_in = len(inputSeq)
        len_gl = len(germlineSeq)
        if regionDefinition is not None:
            L = regionDefinition.seqLength
        else:
            L = max(len_in, len_gl)
        short = min(len_in, len_gl, L)
        c_in = list(inputSeq)[:short]
        c_gl = list(germlineSeq)[:short]
        if short < L:
            fill = ["N"] * (L - short)
            c_in += fill
            c_gl += fill
        for i, ch in enumerate(c_in):
            if ch == "N" or ch not in (_NUC5 + ["."]):
                c_gl[i] = "N"
        s_gl = "".join(c_gl)
    else:
        s_gl = germlineSeq
    c_gl = list(s_gl)

    ncodon = len(s_gl) // 3
    out = np.empty((4, ncodon * 3), dtype=object)
    for k in range(ncodon):
        codon = s_gl[k * 3:k * 3 + 3]
        if codon not in col_idx:
            codon = "NNN"
        j = col_idx[codon]
        # CODON_TABLE rows: 12 entries, reshape to 3 positions x 4 nucs
        # R: byrow=FALSE -> matrix(codonTable[,codon], nrow=4)
        col = [ct_arr[r, j] for r in range(12)]
        for pos in range(3):
            for nuc in range(4):
                out[nuc, k * 3 + pos] = col[pos * 4 + nuc]
    return out, list(c_gl[:ncodon * 3])


def calcExpectedMutations(germlineSeq, inputSeq=None, targetingModel=None,
                          regionDefinition=None, mutationDefinition=None):
    """Expected R/S mutation frequencies for one sequence.

    Faithful port of shazam's ``calcExpectedMutations``.
    """
    if targetingModel is None:
        targetingModel = HH_S5F
    germlineSeq = re.sub(r"[MRWSYKVHDB]", "N", str(germlineSeq).upper())
    if inputSeq is not None:
        inputSeq = str(inputSeq).upper()

    codonTab = (None if mutationDefinition is None
                else mutationDefinition.codonTable)

    targeting = calculateTargeting(germlineSeq, inputSeq, targetingModel,
                                   regionDefinition)
    germ_for_paths = "".join(targeting.columns)
    paths, path_cols = calculateMutationalPaths(
        germ_for_paths, regionDefinition=regionDefinition, codonTable=codonTab)
    # mark non r/s as None
    paths_arr = np.where(np.isin(paths, ["r", "s"]), paths, None)

    if regionDefinition is None:
        rdLength = max(len(germlineSeq),
                       len(inputSeq) if inputSeq else 0)
        regionDefinition = makeNullRegionDefinition(rdLength)

    bounds = np.array(regionDefinition.boundaries)
    tarvals = targeting.values
    result = {}
    for region in regionDefinition.regions:
        for mt in ("r", "s"):
            mask_region = bounds == region
            tr = tarvals[:4][:, mask_region[:tarvals.shape[1]]]
            pcols_mask = bounds[:paths_arr.shape[1]] == region
            pr = paths_arr[:, pcols_mask]
            # align: targeting region and paths region may differ in length;
            # use min
            ml = min(tr.shape[1], pr.shape[1])
            tr2 = tr[:, :ml]
            pr2 = pr[:, :ml]
            sel = (pr2 == mt)
            result[f"{region}_{mt}"] = float(np.nansum(tr2[sel]))
    arr = np.array(list(result.values()), dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    total = np.nansum(arr)
    if total > 0:
        arr = arr / total
    return {k: v for k, v in zip(result.keys(), arr)}


def expectedMutations(db, sequenceColumn="sequence_alignment",
                      germlineColumn="germline_alignment",
                      targetingModel=None, regionDefinition=None,
                      mutationDefinition=None, cloneColumn="clone_id",
                      juncLengthColumn="junction_length"):
    """Calculate expected mutation frequencies for a data frame.

    Faithful port of shazam's ``expectedMutations``.
    """
    if targetingModel is None:
        targetingModel = HH_S5F
    db = db.copy()
    if regionDefinition is not None:
        labels = regionDefinition.labels
    else:
        labels = makeNullRegionDefinition().labels
    out_labels = [f"mu_expected_{lab}" for lab in labels]
    results = {lab: [] for lab in out_labels}
    rd_name = regionDefinition.name if regionDefinition is not None else ""

    seqs = db[sequenceColumn].astype(str).str.upper().tolist()
    germs = db[germlineColumn].astype(str).str.upper().tolist()
    for idx in range(len(db)):
        rd = regionDefinition
        if rd_name in ("IMGT_VDJ_BY_REGIONS", "IMGT_VDJ"):
            from .regions import setRegionBoundaries
            rd = setRegionBoundaries(
                juncLength=db[juncLengthColumn].iloc[idx],
                sequenceImgt=seqs[idx], regionDefinition=regionDefinition)
        eM = calcExpectedMutations(
            germlineSeq=germs[idx], inputSeq=seqs[idx],
            targetingModel=targetingModel, regionDefinition=rd,
            mutationDefinition=mutationDefinition)
        for lab in labels:
            v = eM.get(lab, 0.0)
            results[f"mu_expected_{lab}"].append(
                0.0 if (isinstance(v, float) and np.isnan(v)) else v)
    for lab in out_labels:
        db[lab] = results[lab]
    return db
