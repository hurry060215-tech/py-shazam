"""Core classes and constants for pyshazam.

Faithful port of shazam's RegionDefinition / MutationDefinition S4 classes and
the codon / mutation-type machinery (R/Core.R, R/RegionDefinitions.R,
R/MutationDefinitions.R, R/ConvertNumbering.R).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import _loader
from ._loader import codon_table   # re-export: (array, rownames, colnames)

# ---------------------------------------------------------------------------
# Constants (from shazam sysdata)
# ---------------------------------------------------------------------------
_C = _loader.constants()
NUCLEOTIDES = _C["NUCLEOTIDES"]                       # A C G T N - .
NUCLEOTIDES_AMBIGUOUS = _C["NUCLEOTIDES_AMBIGUOUS"]   # A C G T M R W S Y K V H D B N - .
AMINO_ACIDS = _C["AMINO_ACIDS"]                       # codon -> aa
EXPANDED_AMBIGUOUS_CODONS = _C["EXPANDED_AMBIGUOUS_CODONS"]
VLENGTH = _C["VLENGTH"]
CONST_I = _C["CONST_I"]
BAYESIAN_FITTED = _C["BAYESIAN_FITTED"]
IUPAC_DNA_2 = _C["IUPAC_DNA_2"]

# IUPAC DNA expansion (single ambiguous char -> tuple of unambiguous nucs)
IUPAC_DNA = {
    "A": ["A"], "C": ["C"], "G": ["G"], "T": ["T"],
    "M": ["A", "C"], "R": ["A", "G"], "W": ["A", "T"],
    "S": ["C", "G"], "Y": ["C", "T"], "K": ["G", "T"],
    "V": ["A", "C", "G"], "H": ["A", "C", "T"],
    "D": ["A", "G", "T"], "B": ["C", "G", "T"],
    "N": ["A", "C", "G", "T"],
}

REGION_PALETTE = {
    "cdr": "#377eb8", "fwr": "#e41a1c", "cdr1": "#ff7f00",
    "cdr2": "#a65628", "cdr3": "#a62828", "fwr1": "#4daf4a",
    "fwr2": "#984ea3", "fwr3": "#e41a1c", "fwr4": "#908cff",
}


# ---------------------------------------------------------------------------
# RegionDefinition (R/RegionDefinitions.R)
# ---------------------------------------------------------------------------
class RegionDefinition:
    """Region boundaries of an Ig sequence (port of shazam's S4 class)."""

    def __init__(self, name="", description="", boundaries=None,
                 boundaries_levels=None, seqLength=None, regions=None,
                 labels=None, citation=""):
        self.name = name
        self.description = description
        self.boundaries = list(boundaries) if boundaries is not None else []
        # levels: ordered unique categories
        if boundaries_levels is not None:
            self.boundaries_levels = list(boundaries_levels)
        else:
            seen = []
            for b in self.boundaries:
                if b not in seen:
                    seen.append(b)
            self.boundaries_levels = seen
        self.seqLength = (int(seqLength) if seqLength is not None
                          else len(self.boundaries))
        if regions is not None:
            self.regions = list(regions)
        else:
            self.regions = list(self.boundaries_levels)
        if labels is not None:
            self.labels = list(labels)
        else:
            self.labels = [f"{r}_{m}" for r in self.regions for m in ("r", "s")]
        self.citation = citation

    def __repr__(self):
        return (f"RegionDefinition(name={self.name!r}, "
                f"seqLength={self.seqLength}, regions={self.regions})")


def createRegionDefinition(name="", boundaries=None, description="",
                           citation=""):
    """Create a RegionDefinition object.

    Parameters
    ----------
    name : str
    boundaries : list
        Per-position region labels.
    description, citation : str
    """
    boundaries = list(boundaries) if boundaries is not None else []
    seen = []
    for b in boundaries:
        if b not in seen:
            seen.append(b)
    regions = seen
    labels = [f"{r}_{m}" for r in regions for m in ("r", "s")]
    return RegionDefinition(name=name, description=description,
                            boundaries=boundaries, boundaries_levels=regions,
                            seqLength=len(boundaries), regions=regions,
                            labels=labels, citation=citation)


def makeNullRegionDefinition(regionLength=0):
    """Create an empty (single-region) RegionDefinition of a given length."""
    return createRegionDefinition(
        name="", boundaries=["seq"] * int(regionLength),
        description="", citation="")


def _load_region_def(name):
    d = _loader.region_def_raw(name)
    return RegionDefinition(
        name=d["name"], description=d["description"],
        boundaries=d["boundaries"], boundaries_levels=d["boundaries_levels"],
        seqLength=d["seqLength"], regions=d["regions"], labels=d["labels"],
        citation=d["citation"])


IMGT_V = _load_region_def("IMGT_V")
IMGT_V_BY_CODONS = _load_region_def("IMGT_V_BY_CODONS")
IMGT_V_BY_REGIONS = _load_region_def("IMGT_V_BY_REGIONS")
IMGT_V_BY_SEGMENTS = _load_region_def("IMGT_V_BY_SEGMENTS")
IMGT_VDJ = _load_region_def("IMGT_VDJ")
IMGT_VDJ_BY_REGIONS = _load_region_def("IMGT_VDJ_BY_REGIONS")

IMGT_SCHEMES = {
    "IMGT_V": IMGT_V, "IMGT_V_BY_CODONS": IMGT_V_BY_CODONS,
    "IMGT_V_BY_REGIONS": IMGT_V_BY_REGIONS,
    "IMGT_V_BY_SEGMENTS": IMGT_V_BY_SEGMENTS, "IMGT_VDJ": IMGT_VDJ,
    "IMGT_VDJ_BY_REGIONS": IMGT_VDJ_BY_REGIONS,
}


# ---------------------------------------------------------------------------
# MutationDefinition (R/MutationDefinitions.R)
# ---------------------------------------------------------------------------
class MutationDefinition:
    """Replacement/silent mutation definition (port of shazam's S4 class)."""

    def __init__(self, name="", description="", classes=None,
                 codonTable=None, codon_rownames=None, codon_colnames=None,
                 citation=""):
        self.name = name
        self.description = description
        self.classes = dict(classes) if classes is not None else {}
        self.codonTable = codonTable           # numpy object array
        self.codon_rownames = codon_rownames
        self.codon_colnames = codon_colnames
        self.citation = citation

    def __repr__(self):
        return f"MutationDefinition(name={self.name!r})"


def _codon_to_dict(arr, rownames, colnames):
    """Lookup helper: column codon -> list of 12 cell values."""
    return {c: [arr[i, j] for i in range(arr.shape[0])]
            for j, c in enumerate(colnames)}


def allCodonMuts(codon):
    """Return the 12 codons one nucleotide-change away from ``codon``."""
    nuc = NUCLEOTIDES[:4]
    cc = list(codon)
    out = []
    for pos in range(3):
        for n in nuc:
            mut = cc.copy()
            mut[pos] = n
            out.append("".join(mut))
    return out


def computeCodonTable(aminoAcidClasses=None):
    """Generate a 12x216 codon mutation-type table.

    Returns (array, rownames, colnames). Each cell is 'r', 's' or None.
    """
    nuc = NUCLEOTIDES[:4]
    codons = []
    for a in nuc:
        for b in nuc:
            for c in nuc:
                codons.append(a + b + c)
    table = {}
    for codon in codons:
        muts = allCodonMuts(codon)
        col = []
        for m in muts:
            mt = mutationType(m, codon, aminoAcidClasses=aminoAcidClasses)
            keys = [k for k in ("r", "s", "stop", "na") if mt[k] > 0]
            if len(keys) != 1:
                raise ValueError("computeCodonTable: ambiguous codon type")
            v = keys[0]
            col.append(None if v == "na" else v)
        table[codon] = col
    # codons with N or . -> all NA
    chars = ["N", "A", "C", "G", "T", "."]
    for n1 in chars:
        for n2 in chars:
            for n3 in chars:
                if "N" in (n1, n2, n3) or "." in (n1, n2, n3):
                    table[n1 + n2 + n3] = [None] * 12
    colnames = list(table.keys())
    arr = np.empty((12, len(colnames)), dtype=object)
    for j, c in enumerate(colnames):
        for i in range(12):
            arr[i, j] = table[c][i]
    return arr, [f"Row{i+1}" for i in range(12)], colnames


def createMutationDefinition(name, classes, description="", citation=""):
    """Create a MutationDefinition object with a custom codon table."""
    arr, rn, cn = computeCodonTable(aminoAcidClasses=classes)
    return MutationDefinition(name=name, description=description,
                              classes=classes, codonTable=arr,
                              codon_rownames=rn, codon_colnames=cn,
                              citation=citation)


def _load_mutation_def(name):
    d = _loader.mutation_def_raw(name)
    raw_cls = d["classes"]
    if isinstance(raw_cls, dict):
        classes = dict(raw_cls)
    else:
        classes = {k: v for k, v in zip(d["classes_names"], raw_cls)}
    ct = d["codonTable"]
    vals = ct["values"]
    nrow = len(vals)
    ncol = len(vals[0]) if nrow else 0
    cta = np.empty((nrow, ncol), dtype=object)
    for i in range(nrow):
        for j in range(ncol):
            cta[i, j] = vals[i][j]
    return MutationDefinition(
        name=d["name"], description=d["description"], classes=classes,
        codonTable=cta, codon_rownames=ct["rownames"],
        codon_colnames=ct["colnames"], citation=d["citation"])


CHARGE_MUTATIONS = _load_mutation_def("CHARGE_MUTATIONS")
HYDROPATHY_MUTATIONS = _load_mutation_def("HYDROPATHY_MUTATIONS")
POLARITY_MUTATIONS = _load_mutation_def("POLARITY_MUTATIONS")
VOLUME_MUTATIONS = _load_mutation_def("VOLUME_MUTATIONS")

MUTATION_SCHEMES = {
    "CHARGE_MUTATIONS": CHARGE_MUTATIONS,
    "HYDROPATHY_MUTATIONS": HYDROPATHY_MUTATIONS,
    "POLARITY_MUTATIONS": POLARITY_MUTATIONS,
    "VOLUME_MUTATIONS": VOLUME_MUTATIONS,
}


# ---------------------------------------------------------------------------
# Codon / mutation-type helpers (R/MutationProfiling.R)
# ---------------------------------------------------------------------------
def getCodonNumb(nucPos):
    """1-based codon number containing a 1-based nucleotide position."""
    return math.ceil(nucPos / 3)


def getContextInCodon(nucPos):
    """Position (1..3) within the codon for a 1-based nucleotide position."""
    return (nucPos - 1) % 3 + 1


def getCodonPos(nucPos):
    """Return the three 1-based nucleotide positions of the codon."""
    codonNum = math.ceil(nucPos / 3) * 3
    return [codonNum - 2, codonNum - 1, codonNum]


def getCodonNucs(codonNumb):
    """Return the three 1-based nucleotide positions of codon ``codonNumb``."""
    return getCodonPos(codonNumb * 3)


def mutationType(codonFrom, codonTo, ambiguousMode="eitherOr",
                 aminoAcidClasses=None):
    """Classify a codon->codon change as r/s/stop/na with counts.

    Faithful port of shazam's ``mutationType``.
    """
    tab = {"r": 0, "s": 0, "stop": 0, "na": 0}
    has_gap = any(ch in codonFrom for ch in "-.") or \
        any(ch in codonTo for ch in "-.")
    if has_gap:
        tab["na"] = 1
        return tab

    from_all = EXPANDED_AMBIGUOUS_CODONS.get(codonFrom)
    to_all = EXPANDED_AMBIGUOUS_CODONS.get(codonTo)
    if from_all is None or to_all is None:
        tab["na"] = 1
        return tab

    for cf in from_all:
        for ct in to_all:
            if cf == ct:
                tab["na"] += 1
            else:
                aaf = AMINO_ACIDS.get(cf)
                aat = AMINO_ACIDS.get(ct)
                if aaf is None or aat is None:
                    tab["na"] += 1
                elif aaf == "*" or aat == "*":
                    tab["stop"] += 1
                elif aminoAcidClasses is None:
                    tab["s" if aaf == aat else "r"] += 1
                else:
                    cf_cls = aminoAcidClasses.get(aaf)
                    ct_cls = aminoAcidClasses.get(aat)
                    tab["s" if cf_cls == ct_cls else "r"] += 1

    if len(from_all) > 1 or len(to_all) > 1:
        if ambiguousMode == "eitherOr":
            if tab["na"] > 0:
                tab = {"r": 0, "s": 0, "stop": 0, "na": 1}
            elif tab["s"] > 0:
                tab = {"r": 0, "s": 1, "stop": 0, "na": 0}
            elif tab["r"] > 0:
                tab = {"r": 1, "s": 0, "stop": 0, "na": 0}
            else:
                tab = {"r": 0, "s": 0, "stop": 1, "na": 0}
    return tab


def checkAmbiguousExist(seqs):
    """Return a boolean numpy array marking sequences with ambiguous chars."""
    import re
    pat = re.compile(r"[^atgcnATGCN\-.]")
    return np.array([bool(pat.search(s)) for s in seqs])


def nucs2IUPAC(nucs):
    """Collapse a set of A/C/G/T nucleotides into a single IUPAC character."""
    legal = {"A", "C", "G", "T"}
    if any(n not in legal for n in nucs):
        raise ValueError("Input nucleotides must be one of A, C, G, or T.")
    key = "".join(sorted(set(nucs)))
    return IUPAC_DNA_2[key]


def IUPAC2nucs(code, excludeN=True):
    """Expand a single IUPAC character into its nucleotides."""
    if code not in IUPAC_DNA:
        raise ValueError("Input character must be one of IUPAC DNA codes.")
    if code == "N" and excludeN:
        return [code]
    return list(IUPAC_DNA[code])


# ---------------------------------------------------------------------------
# IMGT/Kabat numbering conversion (R/ConvertNumbering.R)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _convert_num_ref():
    return _loader._load_json("convert_num_ref")


def convertNumbering(locus, frm, to, calls):
    """Convert between IMGT and Kabat numbering schemes."""
    ref = _convert_num_ref()
    from_map = ref[f"{locus}_{frm}"]
    to_map = ref[f"{locus}_{to}"]
    from_map = [str(x) for x in from_map]
    to_map = [str(x) for x in to_map]
    calls = [str(c) for c in calls]
    if not all(c in from_map for c in calls):
        bad = [c for c in calls if c not in from_map]
        raise ValueError("Formatting of following characters does not match "
                         f"reference: {', '.join(bad)}")
    lookup = {}
    for f, t in zip(from_map, to_map):
        lookup.setdefault(f, t)
    counts = {}
    for f in from_map:
        counts[f] = counts.get(f, 0) + 1
    out = []
    for c in calls:
        if counts.get(c, 0) > 1:
            out.append("NA")
        else:
            out.append(lookup[c])
    return out
