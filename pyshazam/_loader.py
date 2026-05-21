"""Internal data loader for bundled shazam constants, matrices and models."""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")


def _load_json(name):
    with open(os.path.join(_DATA_DIR, name + ".json")) as fh:
        return json.load(fh)


def _to_matrix(d):
    """Reconstruct a labelled matrix from a dict with rownames/colnames/values."""
    vals = d["values"]
    arr = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in vals],
        dtype=float,
    )
    return arr, list(d["rownames"]), list(d["colnames"])


@lru_cache(maxsize=1)
def constants():
    """Return shazam global constants as a dict."""
    c = _load_json("constants")
    out = {
        "NUCLEOTIDES": list(c["NUCLEOTIDES"]),
        "NUCLEOTIDES_AMBIGUOUS": list(c["NUCLEOTIDES_AMBIGUOUS"]),
        "VLENGTH": int(c["VLENGTH"]),
        "CONST_I": np.asarray(c["CONST_I"], dtype=float),
        "BAYESIAN_FITTED": np.asarray(c["BAYESIAN_FITTED"], dtype=float),
        "IUPAC_DNA_2": dict(c["IUPAC_DNA_2"]),
    }
    # AMINO_ACIDS: codon -> single-letter amino acid
    aa = c["AMINO_ACIDS"]
    if isinstance(aa, dict):
        out["AMINO_ACIDS"] = dict(aa)
    else:
        out["AMINO_ACIDS"] = {k: v for k, v in
                              zip(c["AMINO_ACIDS_NAMES"], aa)}
    # EXPANDED_AMBIGUOUS_CODONS: codon -> list of unambiguous codons
    eac = {}
    for k, v in c["EXPANDED_AMBIGUOUS_CODONS"].items():
        eac[k] = list(v) if isinstance(v, list) else [v]
    out["EXPANDED_AMBIGUOUS_CODONS"] = eac
    return out


@lru_cache(maxsize=1)
def codon_table():
    """Return the default 12x216 CODON_TABLE as (array, rownames, colnames).

    Cells contain 'r', 's', 'stop' or None (NA).
    """
    d = _load_json("codon_table")
    vals = d["values"]
    nrow, ncol = int(d["nrow"]), int(d["ncol"])
    arr = np.empty((nrow, ncol), dtype=object)
    for i in range(nrow):
        for j in range(ncol):
            arr[i, j] = vals[i * ncol + j]
    rn = d.get("rownames")
    rn = list(rn) if rn else [f"Row{i+1}" for i in range(nrow)]
    return arr, rn, list(d["colnames"])


@lru_cache(maxsize=1)
def _dist_1mer_raw():
    return _load_json("dist_1mer")


@lru_cache(maxsize=1)
def _dist_5mer_raw():
    return _load_json("dist_5mer")


def dist_matrix(name):
    """Return a precomputed distance matrix (array, rownames, colnames)."""
    for raw in (_dist_1mer_raw(), _dist_5mer_raw()):
        if name in raw:
            return _to_matrix(raw[name])
    raise KeyError(f"Unknown distance matrix: {name}")


@lru_cache(maxsize=1)
def nuc_mats():
    """Return CDR_Nuc_Mat and FWR_Nuc_Mat."""
    raw = _load_json("nuc_mats")
    return {k: _to_matrix(v) for k, v in raw.items()}


@lru_cache(maxsize=None)
def targeting_model_raw(name):
    """Return raw dict for a bundled TargetingModel."""
    return _load_json(name)


@lru_cache(maxsize=None)
def mutation_def_raw(name):
    """Return raw dict for a bundled MutationDefinition."""
    return _load_json(name)


@lru_cache(maxsize=None)
def region_def_raw(name):
    """Return raw dict for a bundled RegionDefinition."""
    return _load_json(name)
