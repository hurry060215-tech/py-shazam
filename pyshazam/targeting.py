"""SHM targeting models: substitution / mutability / targeting matrices.

Faithful port of shazam's R/TargetingModels.R.
"""
from __future__ import annotations

import itertools
import re
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import _loader, core
from .core import NUCLEOTIDES, getCodonPos, codon_table as _codon_table_func

_NUC = NUCLEOTIDES[:4]                       # A C G T
_NUC5 = NUCLEOTIDES[:5]                      # A C G T N
CODON_TABLE_ARR, _CT_RN, CODON_TABLE_COLS = _codon_table_func()
_CT_COL_IDX = {c: j for j, c in enumerate(CODON_TABLE_COLS)}


def _words(k, alphabet):
    """All length-k words over ``alphabet`` (R seqinr::words order)."""
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]


_NUC_WORDS4 = _words(4, _NUC)
_NUC_WORDS5 = _words(5, _NUC)
_NUC_5MERS_N = _words(5, _NUC5)


# ---------------------------------------------------------------------------
# TargetingModel
# ---------------------------------------------------------------------------
class TargetingModel:
    """5-mer SHM targeting model (port of shazam's S4 class)."""

    def __init__(self, name="", description="", species="", date="",
                 citation="", substitution=None, sub_rownames=None,
                 sub_colnames=None, mutability=None, mut_names=None,
                 targeting=None, tar_rownames=None, tar_colnames=None,
                 numMutS=np.nan, numMutR=np.nan, source=None):
        self.name = name
        self.description = description
        self.species = species
        self.date = date
        self.citation = citation
        self.substitution = substitution        # 5x3125 array
        self.sub_rownames = sub_rownames
        self.sub_colnames = sub_colnames
        self.mutability = mutability            # length-3125 array
        self.mut_names = mut_names
        self.targeting = targeting              # 5x3125 array
        self.tar_rownames = tar_rownames
        self.tar_colnames = tar_colnames
        self.numMutS = numMutS
        self.numMutR = numMutR
        self.source = source

    @property
    def targeting_df(self):
        """Targeting matrix as a labelled DataFrame."""
        return pd.DataFrame(self.targeting, index=self.tar_rownames,
                            columns=self.tar_colnames)

    @property
    def mutability_series(self):
        """Mutability vector as a labelled Series."""
        return pd.Series(self.mutability, index=self.mut_names)

    def __repr__(self):
        return (f"TargetingModel(name={self.name!r}, "
                f"species={self.species!r})")


class MutabilityModel(np.ndarray):
    """A length-3125 mutability vector with names and provenance metadata."""

    def __new__(cls, values, names=None, source=None, numMutS=np.nan,
                numMutR=np.nan):
        obj = np.asarray(values, dtype=float).view(cls)
        obj.names = list(names) if names is not None else None
        obj.source = source
        obj.numMutS = numMutS
        obj.numMutR = numMutR
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.names = getattr(obj, "names", None)
        self.source = getattr(obj, "source", None)
        self.numMutS = getattr(obj, "numMutS", np.nan)
        self.numMutR = getattr(obj, "numMutR", np.nan)

    def as_series(self):
        """Return the mutability values as a labelled pandas Series."""
        return pd.Series(np.asarray(self), index=self.names)


def _load_targeting_model(name):
    d = _loader.targeting_model_raw(name)
    sub, srn, scn = _loader._to_matrix(d["substitution"])
    tar, trn, tcn = _loader._to_matrix(d["targeting"])
    mut_names = list(d["mutability_names"])
    mut_raw = d["mutability"]
    if isinstance(mut_raw, dict):
        mut = np.array([np.nan if mut_raw.get(k) is None
                        else float(mut_raw[k]) for k in mut_names])
    else:
        mut = np.array([np.nan if v is None else float(v)
                        for v in mut_raw])
    return TargetingModel(
        name=d["name"], description=d["description"], species=d["species"],
        date=d["date"], citation=d["citation"], substitution=sub,
        sub_rownames=srn, sub_colnames=scn, mutability=mut,
        mut_names=mut_names, targeting=tar, tar_rownames=trn,
        tar_colnames=tcn)


HH_S5F = _load_targeting_model("HH_S5F")
MK_RS5NF = _load_targeting_model("MK_RS5NF")
HKL_S5F = _load_targeting_model("HKL_S5F")
U5N = _load_targeting_model("U5N")

# 1-mer substitution matrices (4x4)
_hh1, _hh1_rn, _hh1_cn = _loader._to_matrix(_loader.targeting_model_raw("HH_S1F"))
HH_S1F = pd.DataFrame(_hh1, index=_hh1_rn, columns=_hh1_cn)
_mk1, _mk1_rn, _mk1_cn = _loader._to_matrix(
    _loader.targeting_model_raw("MK_RS1NF"))
MK_RS1NF = pd.DataFrame(_mk1, index=_mk1_rn, columns=_mk1_cn)
_hkl1, _hkl1_rn, _hkl1_cn = _loader._to_matrix(
    _loader.targeting_model_raw("HKL_S1F"))
HKL_S1F = pd.DataFrame(_hkl1, index=_hkl1_rn, columns=_hkl1_cn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def getFamily(v_call):
    """Extract V-segment family from allele calls (alakazam::getFamily)."""
    out = []
    for v in v_call:
        v = str(v).split(",")[0]
        m = re.match(r"([^-*]+)", v)
        out.append(m.group(1) if m else v)
    return out


def removeCodonGaps(seq_pairs):
    """Remove in-frame '...' IMGT gaps from input/germline sequence pairs.

    ``seq_pairs`` is an iterable of (input, germline) tuples; returns a list
    of (input, germline) tuples with '...' codons stripped.
    """
    out = []
    for s_in, s_gl in seq_pairs:
        cods_in = re.findall(r".{3}", s_in)
        cods_gl = re.findall(r".{3}", s_gl)
        ni, ng = [], []
        for ci, cg in zip(cods_in, cods_gl):
            if ci != "..." and cg != "...":
                ni.append(ci)
                ng.append(cg)
        out.append(("".join(ni), "".join(ng)))
    return out


def canMutateTo(nuc):
    """All four nucleotides with ``nuc`` placed last (R helper)."""
    others = [n for n in _NUC if n != nuc]
    return others + [nuc]


def listObservedMutations(db, sequenceColumn, germlineColumn,
                          multipleMutation="independent", model="rs"):
    """List per-sequence mutation positions and their R/S types.

    Returns a list of dicts: position(1-based) -> 'r'/'s'.
    """
    out = []
    for s, g in zip(db[sequenceColumn], db[germlineColumn]):
        s = str(s).upper()
        g = str(g).upper()
        muts = {}
        L = min(len(s), len(g))
        for i in range(L):
            cs, cg = s[i], g[i]
            if cs != cg and cs in "ACGT" and cg in "ACGT":
                pos = i + 1
                cp = getCodonPos(pos)
                if cp[2] > L:
                    continue
                gl_cod = "".join(g[k - 1] for k in cp)
                in_cod = list(gl_cod)
                in_cod[(pos - 1) % 3] = cs
                in_cod = "".join(in_cod)
                if "N" in gl_cod or any(c not in "ACGT" for c in gl_cod):
                    muts[pos] = None
                    continue
                tab = core.mutationType(gl_cod, in_cod)
                if tab["r"] > 0:
                    muts[pos] = "r"
                elif tab["s"] > 0:
                    muts[pos] = "s"
                else:
                    muts[pos] = None
        out.append(muts)
    return out


# ---------------------------------------------------------------------------
# createSubstitutionMatrix
# ---------------------------------------------------------------------------
def createSubstitutionMatrix(db, model="s", sequenceColumn="sequence_alignment",
                             germlineColumn="germline_alignment_d_mask",
                             vCallColumn="v_call",
                             multipleMutation="independent",
                             returnModel="5mer", minNumMutations=50,
                             numMutationsOnly=False):
    """Build a 5-mer (or 1-mer) substitution model.

    Faithful port of shazam's ``createSubstitutionMatrix``.
    """
    db = db.copy()
    db[sequenceColumn] = db[sequenceColumn].astype(str).str.upper()
    db[germlineColumn] = db[germlineColumn].astype(str).str.upper()
    if (core.checkAmbiguousExist(db[sequenceColumn]).any() or
            core.checkAmbiguousExist(db[germlineColumn]).any()):
        raise ValueError("Ambiguous characters are not supported.")

    v_families = getFamily(db[vCallColumn])
    pairs = removeCodonGaps(list(zip(db[sequenceColumn], db[germlineColumn])))
    db = db.assign(**{sequenceColumn: [p[0] for p in pairs],
                      germlineColumn: [p[1] for p in pairs]})

    mutations = listObservedMutations(db, sequenceColumn, germlineColumn,
                                      multipleMutation, model)
    nuc_idx = {n: i for i, n in enumerate(_NUC)}
    VLEN = core.VLENGTH

    # substitutionList[v_fam][word] = 4x4 array
    subList = {}
    for vf in set(v_families):
        subList[vf] = {w: np.zeros((4, 4)) for w in _NUC_WORDS4}

    seqs = db[sequenceColumn].tolist()
    germs = db[germlineColumn].tolist()
    for index, idxMut in enumerate(mutations):
        cSeq = seqs[index]
        cGL = germs[index]
        vf = v_families[index]
        positions = [p for p in idxMut if p <= VLEN]
        for position in positions:
            if position - 3 < 0 or position + 2 > len(cGL):
                continue
            wrd = cGL[position - 3:position - 1] + cGL[position:position + 2]
            cp = getCodonPos(position)
            codonGL = [cGL[i - 1] for i in cp]
            codonSeq = [cSeq[i - 1] for i in cp]
            muCodonPos = (position - 1) % 3
            seqAt = codonSeq[muCodonPos]
            glAt = codonGL[muCodonPos]
            if "N" in codonGL or "N" in codonSeq:
                continue
            if "N" in wrd or len(wrd) != 4:
                continue
            if model == "s":
                # only count if all permutated mutations at this codon are silent
                cands = canMutateTo(glAt)[:3]
                all_silent = True
                for cand in cands:
                    permCodon = codonGL.copy()
                    permCodon[muCodonPos] = cand
                    tab = core.mutationType("".join(permCodon),
                                            "".join(codonGL))
                    if tab["s"] != 1:
                        all_silent = False
                        break
                if all_silent and glAt in nuc_idx and seqAt in nuc_idx:
                    subList[vf][wrd][nuc_idx[glAt], nuc_idx[seqAt]] += 1
            else:  # "rs"
                if glAt in nuc_idx and seqAt in nuc_idx:
                    subList[vf][wrd][nuc_idx[glAt], nuc_idx[seqAt]] += 1

    # Aggregate across V families -> M[word] = 4x4
    M = {}
    subMat1mer = np.zeros((4, 4))
    for word in _NUC_WORDS4:
        agg = np.zeros((4, 4))
        for vf in subList:
            agg += subList[vf][word]
        M[word] = agg
        subMat1mer += agg

    if returnModel == "1mer":
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = subMat1mer / subMat1mer.sum(axis=1, keepdims=True)
        return pd.DataFrame(norm, index=_NUC, columns=_NUC)
    if returnModel == "1mer_raw":
        return pd.DataFrame(subMat1mer, index=_NUC, columns=_NUC)

    # 5-mer model
    def _simplifivemer(FIVEMER, thresh, count):
        nuc = FIVEMER[2]
        nuc_i = nuc_idx[nuc]
        nei = FIVEMER[:2] + FIVEMER[3:]
        FIVE5 = M[nei][nuc_i].copy()
        five_total = FIVE5.sum()
        five_every = (np.sum(FIVE5 == 0) == 1)
        FIVE3 = FIVE5.copy()
        for i in range(4):
            for j in range(4):
                mn = _NUC[i] + nei[1:3] + _NUC[j]
                FIVE3 = FIVE3 + M[mn][nuc_i]
        inner3_total = FIVE3.sum()
        inner3_every = (np.sum(FIVE3 == 0) == 1)
        FIVE1 = FIVE5.copy()
        for mn in _NUC_WORDS4:
            FIVE1 = FIVE1 + M[mn][nuc_i]
        if not count:
            if five_total > thresh and five_every:
                return FIVE5
            if inner3_total > thresh and inner3_every:
                return FIVE3
            return FIVE1
        return (five_total, five_every, inner3_total, inner3_every)

    if not numMutationsOnly:
        cols = _NUC_WORDS5
        mat = np.zeros((4, len(cols)))
        for j, fm in enumerate(cols):
            mat[:, j] = _simplifivemer(fm, minNumMutations, False)
        # assign A->A etc to NA
        for j, fm in enumerate(cols):
            center = fm[2]
            mat[nuc_idx[center], j] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            colsum = np.nansum(mat, axis=0)
            mat = mat / colsum
        mat[~np.isfinite(mat)] = np.nan
        return pd.DataFrame(mat, index=_NUC, columns=cols)
    else:
        rows = []
        for fm in _NUC_WORDS5:
            ft, fe, it, ie = _simplifivemer(fm, minNumMutations, True)
            rows.append({"fivemer.total": ft, "fivemer.every": fe,
                         "inner3.total": it, "inner3.every": ie})
        return pd.DataFrame(rows, index=_NUC_WORDS5)


def minNumMutationsTune(subCount, minNumMutationsRange):
    """Tune ``minNumMutations`` using a numMutationsOnly substitution count."""
    out = {}
    for thresh in minNumMutationsRange:
        ft = subCount["fivemer.total"].values
        fe = subCount["fivemer.every"].values
        it = subCount["inner3.total"].values
        ie = subCount["inner3.every"].values
        c5 = np.sum((ft > thresh) & fe)
        c3 = np.sum(~((ft > thresh) & fe) & ((it > thresh) & ie))
        c1 = np.sum(~((ft > thresh) & fe) & ~((it > thresh) & ie))
        out[thresh] = {"5mer": int(c5), "3mer": int(c3), "1mer": int(c1)}
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# createMutabilityMatrix
# ---------------------------------------------------------------------------
def createMutabilityMatrix(db, substitutionModel, model="s",
                           sequenceColumn="sequence_alignment",
                           germlineColumn="germline_alignment_d_mask",
                           vCallColumn="v_call",
                           multipleMutation="independent",
                           minNumSeqMutations=500, numSeqMutationsOnly=False):
    """Build a 5-mer mutability model.

    Faithful port of shazam's ``createMutabilityMatrix``.
    """
    db = db.copy()
    db[sequenceColumn] = db[sequenceColumn].astype(str).str.upper()
    db[germlineColumn] = db[germlineColumn].astype(str).str.upper()
    if (core.checkAmbiguousExist(db[sequenceColumn]).any() or
            core.checkAmbiguousExist(db[germlineColumn]).any()):
        raise ValueError("Ambiguous characters are not supported.")

    subModel = (substitutionModel.values
                if isinstance(substitutionModel, pd.DataFrame)
                else np.asarray(substitutionModel))
    if subModel.shape != (4, 1024):
        raise ValueError("Please supply a valid 5-mer substitutionModel.")

    pairs = removeCodonGaps(list(zip(db[sequenceColumn], db[germlineColumn])))
    db = db.assign(**{sequenceColumn: [p[0] for p in pairs],
                      germlineColumn: [p[1] for p in pairs]})
    mutations = listObservedMutations(db, sequenceColumn, germlineColumn,
                                      multipleMutation, model)
    VLEN = core.VLENGTH
    nuc_idx = {n: i for i, n in enumerate(_NUC)}
    word_idx = {w: i for i, w in enumerate(_NUC_WORDS5)}

    seqs = db[sequenceColumn].tolist()
    germs = db[germlineColumn].tolist()

    # Foreground counts
    COUNT = []
    for index, idxMut in enumerate(mutations):
        c = np.zeros(1024)
        if not any(v is None for v in idxMut.values()):
            cGL = germs[index]
            cSeq = seqs[index]
            positions = [p for p in idxMut if p <= VLEN]
            for position in positions:
                if position - 3 < 0 or position + 2 > len(cGL):
                    continue
                wrd5 = cGL[position - 3:position + 2]
                if len(wrd5) == 5 and re.fullmatch(r"[ACGT]{5}", wrd5):
                    cp = getCodonPos(position)
                    codonGL = [cGL[i - 1] for i in cp]
                    codonSeq = [cSeq[i - 1] for i in cp]
                    if not any(x in ("N", "-", ".") for x in codonGL) and \
                            not any(x in ("N", "-", ".") for x in codonSeq):
                        c[word_idx[wrd5]] += 1
        COUNT.append(c)

    # substitutionSums: rows = 4 nucs + pairs + triples, cols = 1024
    nuc_words = _NUC_WORDS5
    rows = {}
    for i, n in enumerate(_NUC):
        rows[n] = subModel[i]
    for combo in itertools.combinations(range(4), 2):
        key = "".join(_NUC[k] for k in combo)
        rows[key] = np.nansum(subModel[list(combo)], axis=0)
    for combo in itertools.combinations(range(4), 3):
        key = "".join(_NUC[k] for k in combo)
        rows[key] = np.nansum(subModel[list(combo)], axis=0)

    # Background counts
    sSeqVec = [s.replace(".", "N") for s in seqs]
    sGermVec = [g.replace(".", "N") for g in germs]
    BG_COUNT = []
    for index in range(len(mutations)):
        tmp = np.zeros((VLEN, 1024))
        sGL = sGermVec[index]
        cSeq = sSeqVec[index]
        cGL = sGL[:VLEN]
        for pos in range(3, len(cGL) - 1):     # 1-based 3..len-2
            wrd5 = sGL[pos - 3:pos + 2]
            if len(wrd5) == 5 and re.fullmatch(r"[ACGT]{5}", wrd5):
                cp = getCodonPos(pos)
                if cp[2] > len(cGL) or cp[2] > len(cSeq):
                    continue
                codonGL = [cGL[i - 1] for i in cp]
                codonSeq = [cSeq[i - 1] for i in cp]
                muCodonPos = (pos - 1) % 3
                glAt = codonGL[muCodonPos]
                if not any(x in ("N", "-") for x in codonGL) and \
                        not any(x in ("N", "-") for x in codonSeq):
                    codon_str = "".join(codonGL)
                    col_j = _CT_COL_IDX.get(codon_str)
                    if col_j is None:
                        continue
                    muType = [CODON_TABLE_ARR[k + 4 * muCodonPos, col_j]
                              for k in range(4)]
                    if model == "s":
                        muChars = [_NUC[k] for k in range(4)
                                   if _NUC[k] != glAt and muType[k] == "s"]
                    else:
                        muChars = [_NUC[k] for k in range(4)
                                   if _NUC[k] != glAt]
                    if muChars:
                        key = "".join(muChars)
                        tmp[pos - 1, word_idx[wrd5]] = rows[key][
                            word_idx[wrd5]]
        bg = tmp.sum(axis=0)
        bg[bg == 0] = np.nan
        BG_COUNT.append(bg)

    MutabilityMatrix = np.full((1024, len(mutations)), np.nan)
    MutabilityWeights = np.zeros(len(mutations))
    for i in range(len(mutations)):
        with np.errstate(invalid="ignore", divide="ignore"):
            mm = COUNT[i] / BG_COUNT[i]
            mm = mm / np.nansum(mm)
        mm[~np.isfinite(mm)] = np.nan
        MutabilityMatrix[:, i] = mm
        MutabilityWeights[i] = len(mutations[i])

    # total S/R counts
    totS = sum(sum(1 for v in m.values() if v == "s") for m in mutations)
    totR = sum(sum(1 for v in m.values() if v == "r") for m in mutations)

    # weighted mean
    Mutability_Mean = np.full(1024, np.nan)
    for i in range(1024):
        row = MutabilityMatrix[i]
        valid = ~np.isnan(row)
        w = MutabilityWeights[valid]
        if w.sum() > 0:
            Mutability_Mean[i] = np.average(row[valid], weights=w)
    Mutability_Mean[~np.isfinite(Mutability_Mean)] = np.nan
    Mutability_Mean[Mutability_Mean == 0] = np.nan

    NumSeqMutations = np.array(
        [MutabilityWeights[~np.isnan(MutabilityMatrix[i])].sum()
         for i in range(1024)])
    if numSeqMutationsOnly:
        return pd.Series(NumSeqMutations, index=_NUC_WORDS5)

    Mutability_Mean[NumSeqMutations <= minNumSeqMutations] = np.nan
    mm_dict = {w: Mutability_Mean[i] for i, w in enumerate(_NUC_WORDS5)}

    def _fillHot(FIVEMER, mutability):
        v = mutability.get(FIVEMER)
        if v is not None and not np.isnan(v) and v >= 0.0:
            return v
        nuc = FIVEMER[2]
        FIVE, COUNT_ = 0.0, 0
        if nuc in ("A", "T"):
            for i in range(3):
                for j in range(3):
                    mn = (canMutateTo(FIVEMER[0])[i] + FIVEMER[1:4]
                          + canMutateTo(FIVEMER[4])[j])
                    mv = mutability.get(mn)
                    if mv is not None and not np.isnan(mv):
                        FIVE += mv
                        COUNT_ += 1
            return FIVE / COUNT_ if COUNT_ else np.nan
        if nuc == "G":
            for i in range(3):
                for j in range(3):
                    mn = (canMutateTo(FIVEMER[0])[i]
                          + canMutateTo(FIVEMER[1])[j] + FIVEMER[2:5])
                    mv = mutability.get(mn)
                    if mv is not None and not np.isnan(mv):
                        FIVE += mv
                        COUNT_ += 1
            return FIVE / COUNT_ if COUNT_ else np.nan
        if nuc == "C":
            for i in range(3):
                for j in range(3):
                    mn = (FIVEMER[0:3] + canMutateTo(FIVEMER[3])[i]
                          + canMutateTo(FIVEMER[4])[j])
                    mv = mutability.get(mn)
                    if mv is not None and not np.isnan(mv):
                        FIVE += mv
                        COUNT_ += 1
            return FIVE / COUNT_ if COUNT_ else np.nan
        return np.nan

    complete = {w: _fillHot(w, mm_dict) for w in _NUC_WORDS5}
    for w in [k for k, v in complete.items() if np.isnan(v)]:
        complete[w] = _fillHot(w, complete)
    for w in [k for k, v in complete.items()
              if not np.isnan(v) and v < 1e-6]:
        complete[w] = _fillHot(w, complete)
    for w in list(complete):
        if np.isnan(complete[w]):
            complete[w] = 0.0

    arr = np.array([complete[w] for w in _NUC_WORDS5])
    arr = arr / np.nansum(arr)
    source = {w: ("Inferred" if np.isnan(mm_dict[w]) else "Measured")
              for w in _NUC_WORDS5}
    return MutabilityModel(arr, names=_NUC_WORDS5, source=source,
                           numMutS=totS, numMutR=totR)


def minNumSeqMutationsTune(mutCount, minNumSeqMutationsRange):
    """Tune ``minNumSeqMutations`` using a numSeqMutationsOnly count vector."""
    mc = np.asarray(mutCount)
    out = {}
    for thresh in minNumSeqMutationsRange:
        out[thresh] = {"measured": int(np.sum(mc > thresh)),
                       "inferred": int(np.sum(mc <= thresh))}
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# extend / targeting matrix
# ---------------------------------------------------------------------------
def extendSubstitutionMatrix(substitutionModel):
    """Extend a 4x1024 substitution model to 5x3125 (adding Ns)."""
    if isinstance(substitutionModel, pd.DataFrame):
        cols = list(substitutionModel.columns)
        sub = substitutionModel.values
    else:
        cols = _NUC_WORDS4
        sub = np.asarray(substitutionModel)
    col_idx = {c: j for j, c in enumerate(cols)}
    ext = np.full((5, len(_NUC_5MERS_N)), np.nan)
    for j, mer in enumerate(_NUC_5MERS_N):
        if mer in col_idx:
            ext[:4, j] = sub[:, col_idx[mer]]
            ext[4, j] = np.nan
        else:
            mer_char = list(mer)
            n_pos = [k for k, ch in enumerate(mer_char) if ch == "N"]
            if 2 in n_pos:
                ext[:, j] = np.nan
            else:
                for k in n_pos:
                    mer_char[k] = "."
                pat = "".join(mer_char)
                regex = re.compile(pat)
                matched = [col_idx[c] for c in cols if regex.search(c)]
                ext[:4, j] = np.nanmean(sub[:, matched], axis=1)
                ext[4, j] = np.nan
    ext[~np.isfinite(ext)] = np.nan
    return pd.DataFrame(ext, index=_NUC5, columns=_NUC_5MERS_N)


def extendMutabilityMatrix(mutabilityModel):
    """Extend a length-1024 mutability model to length-3125 (adding Ns)."""
    if isinstance(mutabilityModel, MutabilityModel):
        names = mutabilityModel.names
        vals = np.asarray(mutabilityModel)
        numMutS = mutabilityModel.numMutS
        numMutR = mutabilityModel.numMutR
        src = mutabilityModel.source
    elif isinstance(mutabilityModel, pd.Series):
        names = list(mutabilityModel.index)
        vals = mutabilityModel.values
        numMutS = numMutR = np.nan
        src = None
    else:
        names = _NUC_WORDS5
        vals = np.asarray(mutabilityModel)
        numMutS = numMutR = np.nan
        src = None
    name_idx = {n: i for i, n in enumerate(names)}
    ext = np.full(len(_NUC_5MERS_N), np.nan)
    for j, mer in enumerate(_NUC_5MERS_N):
        if mer in name_idx:
            ext[j] = vals[name_idx[mer]]
        else:
            mer_char = list(mer)
            n_pos = [k for k, ch in enumerate(mer_char) if ch == "N"]
            if 2 in n_pos:
                ext[j] = np.nan
            else:
                for k in n_pos:
                    mer_char[k] = "."
                pat = "".join(mer_char)
                regex = re.compile(pat)
                matched = [name_idx[n] for n in names if regex.search(n)]
                if matched:
                    ext[j] = np.nanmean(vals[list(matched)])
    ext[~np.isfinite(ext)] = np.nan
    new_src = None
    if src is not None:
        new_src = {n: "Extended" for n in _NUC_5MERS_N}
        for n, v in src.items():
            if n in new_src:
                new_src[n] = v
    return MutabilityModel(ext, names=_NUC_5MERS_N, source=new_src,
                           numMutS=numMutS, numMutR=numMutR)


def createTargetingMatrix(substitutionModel, mutabilityModel):
    """Combine substitution and mutability into a targeting matrix."""
    sub = (substitutionModel.values
           if isinstance(substitutionModel, pd.DataFrame)
           else np.asarray(substitutionModel))
    mut = np.asarray(mutabilityModel)
    tar = sub * mut[np.newaxis, :]
    tar[~np.isfinite(tar)] = np.nan
    cols = (list(substitutionModel.columns)
            if isinstance(substitutionModel, pd.DataFrame) else _NUC_5MERS_N)
    return pd.DataFrame(tar, index=_NUC5, columns=cols)


def createTargetingModel(db, model="s", sequenceColumn="sequence_alignment",
                         germlineColumn="germline_alignment_d_mask",
                         vCallColumn="v_call",
                         multipleMutation="independent", minNumMutations=50,
                         minNumSeqMutations=500, modelName="",
                         modelDescription="", modelSpecies="",
                         modelCitation="", modelDate=None):
    """Build a full 5-mer ``TargetingModel`` from sequence data."""
    if modelDate is None:
        modelDate = date.today().isoformat()
    sub_mat = createSubstitutionMatrix(
        db, model=model, sequenceColumn=sequenceColumn,
        germlineColumn=germlineColumn, vCallColumn=vCallColumn,
        multipleMutation=multipleMutation, minNumMutations=minNumMutations,
        returnModel="5mer")
    mut_mat = createMutabilityMatrix(
        db, sub_mat, model=model, sequenceColumn=sequenceColumn,
        germlineColumn=germlineColumn, vCallColumn=vCallColumn,
        multipleMutation=multipleMutation,
        minNumSeqMutations=minNumSeqMutations)
    sub_ext = extendSubstitutionMatrix(sub_mat)
    mut_ext = extendMutabilityMatrix(mut_mat)
    tar = createTargetingMatrix(sub_ext, mut_ext)
    return TargetingModel(
        name=modelName, description=modelDescription, species=modelSpecies,
        date=modelDate, citation=modelCitation, substitution=sub_ext.values,
        sub_rownames=list(sub_ext.index), sub_colnames=list(sub_ext.columns),
        mutability=np.asarray(mut_ext), mut_names=mut_ext.names,
        targeting=tar.values, tar_rownames=list(tar.index),
        tar_colnames=list(tar.columns), numMutS=mut_ext.numMutS,
        numMutR=mut_ext.numMutR, source=mut_ext.source)


# ---------------------------------------------------------------------------
# calcTargetingDistance / symmetrize / calculateMutability
# ---------------------------------------------------------------------------
def symmetrize(sub1mer):
    """Symmetrize a 4x4 1-mer substitution matrix (port of shazam helper)."""
    if isinstance(sub1mer, pd.DataFrame):
        rn = [r.upper() for r in sub1mer.index]
        cn = [c.upper() for c in sub1mer.columns]
        m = sub1mer.values
    else:
        rn, cn = list("ACGT"), list("ACGT")
        m = np.asarray(sub1mer)
    ri = {r: i for i, r in enumerate(rn)}
    ci = {c: i for i, c in enumerate(cn)}

    def g(a, b):
        return m[ri[a], ci[b]]

    def minDist(pars):
        return ((pars[0] - g("A", "C"))**2 + (pars[0] - g("C", "A"))**2 +
                (pars[0] - g("G", "T"))**2 + (pars[0] - g("T", "G"))**2 +
                (pars[1] - g("A", "G"))**2 + (pars[1] - g("G", "A"))**2 +
                (pars[1] - g("C", "T"))**2 + (pars[1] - g("T", "C"))**2 +
                (pars[2] - g("A", "T"))**2 + (pars[2] - g("T", "A"))**2 +
                (pars[2] - g("C", "G"))**2 + (pars[2] - g("G", "C"))**2)

    res = minimize(minDist, np.zeros(3), method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
    pars = res.x
    pars = pars / pars.sum()
    sym = m.copy().astype(float)
    # rows ordered A,C,G,T
    sym[0, 1:4] = pars                       # A -> C,G,T
    sym[1, [0, 3, 2]] = pars                 # C -> A,T,G
    sym[2, [3, 0, 1]] = pars                 # G -> T,A,C
    sym[3, [2, 1, 0]] = pars                 # T -> G,C,A
    np.fill_diagonal(sym, np.nan)
    return pd.DataFrame(sym, index=list("ACGT"), columns=list("ACGT"))


def calcTargetingDistance(model, places=2):
    """Compute a nucleotide distance matrix from a targeting model.

    Faithful port of shazam's ``calcTargetingDistance``.
    """
    if isinstance(model, TargetingModel):
        inp = "5mer"
        cols = list(model.tar_colnames)
        rows = list(model.tar_rownames)
        mat = model.targeting.copy()
    elif (isinstance(model, (pd.DataFrame, np.ndarray)) and
          np.asarray(model).shape == (4, 4)):
        inp = "1mer"
        sym = symmetrize(model)
        mat = sym.values.copy()
        rows = list(sym.index)
        cols = list(sym.columns)
    else:
        raise ValueError("Input must be a 4x4 matrix or TargetingModel.")

    with np.errstate(invalid="ignore", divide="ignore"):
        dist = -np.log10(mat)
        dist = dist / np.nanmean(dist)
    dist[~np.isfinite(dist)] = np.nan

    if inp == "5mer":
        center = [c[2] for c in cols]
        row_idx = {r: i for i, r in enumerate(rows)}
        for j, c in enumerate(center):
            dist[row_idx[c], j] = 0.0
        for j, c in enumerate(center):
            if c == "N":
                dist[:, j] = 0.0
        dist[row_idx["N"], :] = 0.0
    else:
        np.fill_diagonal(dist, 0.0)
        dist = np.vstack([dist, np.zeros((3, 4))])
        dist = np.hstack([dist, np.zeros((7, 3))])
        rows = list("ACGT") + ["N", "-", "."]
        cols = list("ACGT") + ["N", "-", "."]

    dist = np.round(dist, places)
    return pd.DataFrame(dist, index=rows, columns=cols)


def calculateMutability(sequences, model=None, progress=False):
    """Total (summed) mutability for a set of sequences."""
    if model is None:
        model = HH_S5F
    model_kmer = list(model.mut_names)
    rates = {k: v for k, v in zip(model_kmer, model.mutability)}
    out = []
    for s in sequences:
        s = str(s).upper().replace(".", "N")
        total = 0.0
        for i in range(len(s) - 4):
            kmer = s[i:i + 5]
            r = rates.get(kmer)
            if r is not None and not np.isnan(r):
                total += r
        out.append(total)
    return np.array(out)


def rescaleMutability(model, mean=1.0):
    """Rescale a mutability vector to a given mean."""
    if isinstance(model, TargetingModel):
        vals = np.asarray(model.mutability, dtype=float)
    else:
        vals = np.asarray(model, dtype=float)
    n_valid = np.sum(~np.isnan(vals))
    rescaled = vals / np.nansum(vals) * n_valid * mean
    rescaled[~np.isfinite(rescaled)] = np.nan
    return rescaled


def writeTargetingDistance(model, file):
    """Write a 5-mer targeting distance matrix to a tab-delimited file."""
    df = calcTargetingDistance(model)
    df = df.fillna(0)
    df.to_csv(file, sep="\t")
