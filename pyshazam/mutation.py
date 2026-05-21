"""Mutation analysis: observed / expected mutation counting and sliding window.

Faithful port of shazam's R/MutationProfiling.R.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import core
from .core import (NUCLEOTIDES, NUCLEOTIDES_AMBIGUOUS, EXPANDED_AMBIGUOUS_CODONS,
                   getCodonPos, getContextInCodon, mutationType,
                   makeNullRegionDefinition)

_NUC14 = NUCLEOTIDES_AMBIGUOUS[:14]   # all valid nucleotide chars (no N/-/.)


def _clean_imgt_gaps(seq):
    """Replace single dots with N, keep triplet IMGT gaps as '...'."""
    seq = seq.replace("...", "ZZZ")
    seq = seq.replace(".", "N")
    seq = seq.replace("ZZZ", "...")
    return seq


# ---------------------------------------------------------------------------
# calcObservedMutations
# ---------------------------------------------------------------------------
def calcObservedMutations(inputSeq, germlineSeq, regionDefinition=None,
                          mutationDefinition=None, ambiguousMode="eitherOr",
                          returnRaw=False, frequency=False):
    """Count R/S mutations between an observed and a germline sequence.

    Faithful port of shazam's ``calcObservedMutations``.

    Returns
    -------
    dict or list
        If ``returnRaw=False``: dict keyed by region labels (``cdr_r`` ...).
        If ``returnRaw=True``: ``{"pos": DataFrame|None, "nonN": dict}``.
    """
    if inputSeq is None or (isinstance(inputSeq, float) and np.isnan(inputSeq)):
        inputSeq = ""
    if germlineSeq is None or (isinstance(germlineSeq, float)
                               and np.isnan(germlineSeq)):
        inputSeq = ""

    inputSeq = str(inputSeq).upper()
    germlineSeq = str(germlineSeq).upper()

    aaClasses = (None if mutationDefinition is None
                 else mutationDefinition.classes)

    germlineSeq = _clean_imgt_gaps(germlineSeq)
    inputSeq = _clean_imgt_gaps(inputSeq)

    len_in = len(inputSeq)
    len_gl = len(germlineSeq)

    if regionDefinition is not None:
        rdLength = regionDefinition.seqLength
    else:
        rdLength = max(len_in, len_gl)
        regionDefinition = makeNullRegionDefinition(rdLength)

    len_shortest = min(len_in, len_gl, rdLength)
    c_input = list(inputSeq)[:len_shortest]
    c_germ = list(germlineSeq)[:len_shortest]

    if len_shortest < rdLength:
        fill = ["N"] * (rdLength - len_shortest)
        c_input += fill
        c_germ += fill

    seqLen = len(c_input)
    tooShort = seqLen < 3
    if not tooShort and (seqLen % 3) != 0:
        keep = seqLen - (seqLen % 3)
        c_input = c_input[:keep]
        c_germ = c_germ[:keep]

    labels = regionDefinition.labels
    mutations_array = {lab: np.nan for lab in labels}
    mutations_array_raw = None    # dict: row -> {"r":, "s":, ...}
    mutations_pos = []
    c_input_codons = None
    c_germ_codons = None

    if not tooShort:
        c_input = np.array(c_input)
        c_germ = np.array(c_germ)
        valid_in = np.isin(c_input, _NUC14)
        valid_gl = np.isin(c_germ, _NUC14)
        mutations = (c_germ != c_input) & valid_in & valid_gl
        if mutations.sum() > 0:
            mut_pos = np.where(mutations)[0] + 1   # 1-based
            mutations_pos = list(mut_pos)
            # extract codons
            gl_codon_chars = []
            in_codon_chars = []
            for p in mut_pos:
                cp = getCodonPos(int(p))         # 1-based positions
                gl_cod = [c_germ[i - 1] for i in cp]
                in_cod = gl_cod.copy()
                in_cod[getContextInCodon(int(p)) - 1] = c_input[p - 1]
                gl_codon_chars.extend(gl_cod)
                in_codon_chars.extend(in_cod)
            c_germ_codons = ["".join(gl_codon_chars[i:i + 3])
                             for i in range(0, len(gl_codon_chars), 3)]
            c_input_codons = ["".join(in_codon_chars[i:i + 3])
                              for i in range(0, len(in_codon_chars), 3)]

            raw = {}
            for i, p in enumerate(mut_pos):
                tab = mutationType(c_germ_codons[i], c_input_codons[i],
                                   ambiguousMode=ambiguousMode,
                                   aminoAcidClasses=aaClasses)
                raw[int(p)] = tab
            # keep only positions with r or s
            kept = {p: t for p, t in raw.items() if t["r"] > 0 or t["s"] > 0}
            if not kept:
                mutations_array_raw = None
                mutations_array = {lab: np.nan for lab in labels}
            else:
                mutations_array_raw = kept
                mutations_array = binMutationsByRegion(kept, regionDefinition)

    if frequency:
        if mutations_array_raw is None:
            return mutations_array
        denoms = countNonNByRegion(
            regionDefinition, ambiguousMode, c_input, c_germ,
            c_input_codons, c_germ_codons, mutations_pos)
        out = {}
        for r in regionDefinition.regions:
            d = denoms.get(r, 0)
            for m in ("r", "s"):
                lab = f"{r}_{m}"
                v = mutations_array[lab]
                out[lab] = (v / d) if (d and not np.isnan(v)) else (
                    np.nan if d == 0 else v / d)
        return out

    if returnRaw:
        if mutations_array_raw is None:
            if not tooShort:
                nonN = countNonNByRegion(regionDefinition, ambiguousMode,
                                         c_input, c_germ, None, None, None)
            else:
                nonN = {r: np.nan for r in regionDefinition.regions}
            return {"pos": None, "nonN": nonN}
        nonN = countNonNByRegion(regionDefinition, ambiguousMode, c_input,
                                 c_germ, c_input_codons, c_germ_codons,
                                 mutations_pos)
        rows = []
        for p in sorted(mutations_array_raw):
            t = mutations_array_raw[p]
            reg = regionDefinition.boundaries[p - 1] \
                if p - 1 < len(regionDefinition.boundaries) else None
            rows.append({"position": p, "r": t["r"], "s": t["s"],
                         "region": reg})
        return {"pos": pd.DataFrame(rows), "nonN": nonN}

    return mutations_array


def binMutationsByRegion(mutationsArray, regionDefinition=None):
    """Aggregate R/S mutation counts by region.

    ``mutationsArray`` is a dict: position -> {"r":, "s":, ...}.
    """
    positions = sorted(mutationsArray)
    if regionDefinition is None:
        regionDefinition = makeNullRegionDefinition(max(positions))
    seqLen = regionDefinition.seqLength
    mut_R = np.full(seqLen, np.nan)
    mut_S = np.full(seqLen, np.nan)
    for p in positions:
        if p - 1 < seqLen:
            mut_R[p - 1] = mutationsArray[p]["r"]
            mut_S[p - 1] = mutationsArray[p]["s"]
    counts = {lab: 0 for lab in regionDefinition.labels}
    bounds = np.array(regionDefinition.boundaries[:seqLen])
    for reg in regionDefinition.regions:
        mask = bounds == reg
        counts[f"{reg}_r"] = float(np.nansum(mut_R[mask]))
        counts[f"{reg}_s"] = float(np.nansum(mut_S[mask]))
    return counts


def countNonNByRegion(regDef, ambiMode, inputChars, germChars,
                      inputCodons, germCodons, mutPos):
    """Count non-N/dash/dot positions by region (port of shazam helper)."""
    regionNames = []
    for lab in regDef.labels:
        rn = lab[:-2]
        if rn not in regionNames:
            regionNames.append(rn)
    inputChars = np.asarray(inputChars)
    germChars = np.asarray(germChars)
    bounds = np.array(regDef.boundaries)

    if ambiMode == "eitherOr":
        mask = np.isin(inputChars, _NUC14) & np.isin(germChars, _NUC14)
        sel = bounds[:len(mask)][mask]
        return {x: int(np.sum(sel == x)) for x in regionNames}

    # "and" mode
    mask1 = (np.isin(inputChars, _NUC14) & np.isin(germChars, _NUC14)
             & (germChars == inputChars))
    sel1 = bounds[:len(mask1)][mask1]
    nonN1 = {x: int(np.sum(sel1 == x)) for x in regionNames}
    nonN2 = {x: 0 for x in regionNames}
    if inputCodons is not None and germCodons is not None and mutPos is not None:
        in_exp = np.array([len(EXPANDED_AMBIGUOUS_CODONS.get(c, [c]))
                           for c in inputCodons])
        gl_exp = np.array([len(EXPANDED_AMBIGUOUS_CODONS.get(c, [c]))
                           for c in germCodons])
        total = in_exp * gl_exp
        bnd2 = np.array([regDef.boundaries[p - 1] for p in mutPos])
        for x in regionNames:
            nonN2[x] = int(np.sum(total[bnd2 == x]))
    return {x: nonN1[x] + nonN2[x] for x in regionNames}


# ---------------------------------------------------------------------------
# observedMutations (data-frame interface)
# ---------------------------------------------------------------------------
def observedMutations(db, sequenceColumn="sequence_alignment",
                      germlineColumn="germline_alignment_d_mask",
                      regionDefinition=None, mutationDefinition=None,
                      ambiguousMode="eitherOr", frequency=False,
                      combine=False, cloneColumn="clone_id",
                      juncLengthColumn="junction_length"):
    """Count observed mutations for all sequences in a data frame.

    Faithful port of shazam's ``observedMutations``.
    """
    db = db.copy()
    if regionDefinition is not None:
        labels = regionDefinition.labels
    else:
        labels = makeNullRegionDefinition().labels

    prefix = "mu_freq" if frequency else "mu_count"
    if combine:
        out_labels = [prefix]
    else:
        out_labels = [f"{prefix}_{lab}" for lab in labels]

    results = {lab: [] for lab in out_labels}
    seqs = db[sequenceColumn].tolist()
    germs = db[germlineColumn].tolist()
    rd_name = regionDefinition.name if regionDefinition is not None else ""

    for idx in range(len(db)):
        rd = regionDefinition
        if rd_name in ("IMGT_VDJ_BY_REGIONS", "IMGT_VDJ"):
            from .regions import setRegionBoundaries
            rd = setRegionBoundaries(
                juncLength=db[juncLengthColumn].iloc[idx],
                sequenceImgt=seqs[idx], regionDefinition=regionDefinition)
        oM = calcObservedMutations(
            seqs[idx], germs[idx],
            frequency=frequency and not combine,
            regionDefinition=rd, mutationDefinition=mutationDefinition,
            returnRaw=combine, ambiguousMode=ambiguousMode)
        if combine:
            if oM["pos"] is None or len(oM["pos"]) == 0:
                num_mut = 0
            else:
                num_mut = float(oM["pos"]["r"].sum() + oM["pos"]["s"].sum())
            if not frequency:
                results[prefix].append(num_mut)
            else:
                num_nonN = sum(v for v in oM["nonN"].values()
                               if not (isinstance(v, float) and np.isnan(v)))
                results[prefix].append(num_mut / num_nonN
                                       if num_nonN else np.nan)
        else:
            for lab in labels:
                v = oM[lab]
                results[f"{prefix}_{lab}"].append(0.0 if (
                    isinstance(v, float) and np.isnan(v)) else v)

    for lab in out_labels:
        db[lab] = results[lab]
    return db


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------
def slideWindowSeqHelper(mutPos, mutThresh, windowSize):
    """Return True if a window with >= mutThresh mutations exists."""
    if not (1 <= mutThresh <= windowSize and windowSize >= 2):
        raise ValueError("Invalid mutThresh / windowSize.")
    if mutPos is None or (hasattr(mutPos, "__len__") and len(mutPos) == 0):
        return False
    positions = sorted(int(p) for p in mutPos["position"])
    n = len(positions)
    if n < mutThresh:
        return False
    for i in range(n - mutThresh + 1):
        if positions[i + mutThresh - 1] - positions[i] < windowSize:
            return True
    return False


def slideWindowSeq(inputSeq, germlineSeq, mutThresh, windowSize):
    """Sliding-window filter for a single sequence pair."""
    raw = calcObservedMutations(inputSeq, germlineSeq, returnRaw=True)
    return slideWindowSeqHelper(raw["pos"], mutThresh, windowSize)


def slideWindowDb(db, sequenceColumn="sequence_alignment",
                  germlineColumn="germline_alignment_d_mask",
                  mutThresh=6, windowSize=10, nproc=1):
    """Apply the sliding-window filter to every row of ``db``."""
    db = db.copy()
    flags = []
    for s, g in zip(db[sequenceColumn], db[germlineColumn]):
        flags.append(slideWindowSeq(s, g, mutThresh, windowSize))
    db["FILTERED"] = flags
    return db


def slideWindowTune(db, sequenceColumn="sequence_alignment",
                    germlineColumn="germline_alignment_d_mask",
                    dbMutList=None, mutThreshRange=None, windowSizeRange=None,
                    verbose=True, nproc=1):
    """Parameter tuning for the sliding-window approach.

    Returns a list of boolean numpy arrays, one per (mutThresh, windowSize)
    combination, marking which sequences would be filtered.
    """
    if mutThreshRange is None:
        mutThreshRange = list(range(2, 9))
    if windowSizeRange is None:
        windowSizeRange = list(range(7, 16))
    if dbMutList is None:
        dbMutList = [calcObservedMutations(s, g, returnRaw=True)["pos"]
                     for s, g in zip(db[sequenceColumn], db[germlineColumn])]
    out = {}
    for w in windowSizeRange:
        for m in mutThreshRange:
            if m > w:
                continue
            out[(m, w)] = np.array(
                [slideWindowSeqHelper(p, m, w) for p in dbMutList])
    return out
