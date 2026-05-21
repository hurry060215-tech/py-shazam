"""Clonal consensus / collapsing (R/MutationProfiling.R)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import core
from .core import NUCLEOTIDES_AMBIGUOUS, IUPAC_DNA, IUPAC_DNA_2

_TAB_ROWNAMES = ["A", "T", "G", "C", "N", ".", "-", "na"]


def _chars2Ambiguous(chars):
    """Collapse a set of nucleotide characters into one IUPAC character."""
    chars = list(dict.fromkeys(chars))   # unique, ordered
    legal = {"A", "C", "G", "T", "N", "-", "."}
    if any(c not in legal for c in chars):
        raise ValueError("Illegal character for chars2Ambiguous.")
    if any(c in ("A", "C", "G", "T", "N") for c in chars):
        chars = [c for c in chars if c not in ("-", ".")]
        if all(c == "N" for c in chars):
            return "N"
        chars = [c for c in chars if c != "N"]
        return core.nucs2IUPAC(chars)
    else:
        if set(chars) >= {"-", "."}:
            return "-"
        return chars[0]


def consensusSequence(sequences, db=None, method="mostCommon", minFreq=None,
                      muFreqColumn=None, lenLimit=None, includeAmbiguous=False,
                      breakTiesStochastic=False, breakTiesByColumns=None,
                      rng=None):
    """Build a consensus sequence from a set of aligned sequences.

    Faithful port of shazam's ``consensusSequence``.
    """
    if rng is None:
        rng = np.random.default_rng()
    sequences = [str(s) for s in sequences]
    numSeqs = len(sequences)

    if method in ("mostMutated", "leastMutated"):
        if muFreqColumn is None or db is None:
            raise ValueError("muFreqColumn and db required for most/leastMutated.")
        muFreq = np.asarray(db[muFreqColumn], dtype=float)

    if numSeqs == 1:
        cons = sequences[0]
        if lenLimit is not None:
            cons = cons[:min(lenLimit, len(cons))]
        muf = (muFreq[0] if method in ("mostMutated", "leastMutated")
               else None)
        return {"cons": cons, "muFreq": muf}

    if len(set(sequences)) == 1:
        cons = sequences[0]
        if lenLimit is not None:
            cons = cons[:min(lenLimit, len(cons))]
        muf = (muFreq[0] if method in ("mostMutated", "leastMutated")
               else None)
        return {"cons": cons, "muFreq": muf}

    lenSeqs = [len(s) for s in sequences]
    lenMax = max(lenSeqs)

    if method in ("thresholdedFreq", "mostCommon", "catchAll"):
        if method != "catchAll":
            rownames = _TAB_ROWNAMES
        else:
            rownames = list(NUCLEOTIDES_AMBIGUOUS) + ["na"]
        row_idx = {c: i for i, c in enumerate(rownames)}
        tab = np.zeros((len(rownames), lenMax))
        for j in range(lenMax):
            for i, s in enumerate(sequences):
                if j < lenSeqs[i]:
                    c = s[j]
                    if c not in row_idx:
                        raise ValueError("Unexpected character in sequences.")
                    tab[row_idx[c], j] += 1
                else:
                    tab[row_idx["na"], j] += 1
        numNAs = tab[row_idx["na"]]
        numNonNAs = numSeqs - numNAs
        keep = numNonNAs > (numSeqs // 2)
        lenConsensus = int(np.sum(keep))
        if lenConsensus == 0:
            return {"cons": "", "muFreq": None}
        if lenLimit is not None:
            lenConsensus = min(lenConsensus, lenLimit)
        tab = tab[:, :lenConsensus] / numSeqs
        # remove "na" row
        na_i = row_idx["na"]
        keep_rows = [i for i in range(len(rownames)) if i != na_i]
        sub_rownames = [rownames[i] for i in keep_rows]
        tab = tab[keep_rows]

        cons_chars = []
        for j in range(lenConsensus):
            col = tab[:, j]
            if method == "thresholdedFreq":
                idx = np.where(col >= minFreq)[0]
                if len(idx) == 0:
                    cons_chars.append("N")
                elif len(idx) == 1:
                    cons_chars.append(sub_rownames[idx[0]])
                else:
                    if includeAmbiguous:
                        cons_chars.append(_chars2Ambiguous(
                            [sub_rownames[k] for k in idx]))
                    elif breakTiesStochastic:
                        cons_chars.append(sub_rownames[rng.choice(idx)])
                    else:
                        cons_chars.append(sub_rownames[idx[0]])
            elif method == "mostCommon":
                mx = col.max()
                idx = np.where(np.abs(col - mx) <= 1e-5)[0]
                if len(idx) == 1:
                    cons_chars.append(sub_rownames[idx[0]])
                else:
                    if includeAmbiguous:
                        cons_chars.append(_chars2Ambiguous(
                            [sub_rownames[k] for k in idx]))
                    elif breakTiesStochastic:
                        cons_chars.append(sub_rownames[rng.choice(idx)])
                    else:
                        cons_chars.append(sub_rownames[idx[0]])
            else:  # catchAll
                nz = [sub_rownames[k] for k in range(len(sub_rownames))
                      if col[k] > 0]
                expanded = []
                for c in nz:
                    if c in ("N", "-", "."):
                        expanded.append(c)
                    else:
                        expanded.extend(IUPAC_DNA.get(c, [c]))
                expanded = list(dict.fromkeys(expanded))
                cons_chars.append(_chars2Ambiguous(expanded))
        return {"cons": "".join(cons_chars), "muFreq": None}

    # mostMutated / leastMutated
    if lenLimit is not None:
        sequences = [s[:lenLimit] if len(s) > lenLimit else s
                     for s in sequences]
    if method == "mostMutated":
        target = np.nanmax(muFreq)
    else:
        target = np.nanmin(muFreq)
    idx = np.where(np.abs(muFreq - target) <= 1e-5)[0]
    if len(idx) == 1:
        cons = sequences[idx[0]]
    elif breakTiesStochastic:
        cons = sequences[rng.choice(idx)]
    else:
        cons = sequences[idx[0]]
    return {"cons": cons, "muFreq": target}


def collapseClones(db, cloneColumn="clone_id",
                   sequenceColumn="sequence_alignment",
                   germlineColumn="germline_alignment_d_mask",
                   muFreqColumn=None, regionDefinition=None,
                   method="mostCommon", minimumFrequency=None,
                   includeAmbiguous=False, breakTiesStochastic=False,
                   breakTiesByColumns=None, expandedDb=False,
                   nproc=1, juncLengthColumn="junction_length"):
    """Generate effective clonal consensus sequences.

    Faithful port of shazam's ``collapseClones`` (core methods).
    Adds ``clonal_sequence`` and ``clonal_germline`` columns to ``db``.
    """
    db = db.copy().reset_index(drop=True)
    clonal_seq = [None] * len(db)
    clonal_germ = [None] * len(db)

    for clone, grp in db.groupby(cloneColumn):
        idx = grp.index.tolist()
        seqs = grp[sequenceColumn].astype(str).tolist()
        germs = grp[germlineColumn].astype(str).tolist()
        lenLimit = (regionDefinition.seqLength
                    if regionDefinition is not None else None)
        seq_cons = consensusSequence(
            seqs, db=grp, method=method, minFreq=minimumFrequency,
            muFreqColumn=muFreqColumn, lenLimit=lenLimit,
            includeAmbiguous=includeAmbiguous,
            breakTiesStochastic=breakTiesStochastic,
            breakTiesByColumns=breakTiesByColumns)["cons"]
        # germline consensus: most common at each position
        germ_cons = consensusSequence(
            germs, method="mostCommon", lenLimit=lenLimit)["cons"]
        # trim to same length
        L = min(len(seq_cons), len(germ_cons))
        for i in idx:
            clonal_seq[i] = seq_cons[:L]
            clonal_germ[i] = germ_cons[:L]

    if expandedDb:
        db["clonal_sequence"] = clonal_seq
        db["clonal_germline"] = clonal_germ
        return db
    # collapsed: one row per clone
    rows = []
    for clone, grp in db.groupby(cloneColumn):
        i = grp.index[0]
        row = grp.iloc[0].to_dict()
        row["clonal_sequence"] = clonal_seq[i]
        row["clonal_germline"] = clonal_germ[i]
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)
