"""SHM simulation (R/Shmulate.R): shmulateSeq and shmulateTree."""
from __future__ import annotations

import numpy as np

from . import core
from .core import getCodonPos
from .expected import calculateTargeting
from .targeting import HH_S5F, CODON_TABLE_ARR, CODON_TABLE_COLS

_NUC = core.NUCLEOTIDES         # A C G T N - .
_CT_COL_IDX = {c: j for j, c in enumerate(CODON_TABLE_COLS)}


def computeMutationTypes(inputSeq):
    """Per-position mutation-type matrix (4 x len) from the codon table."""
    L = len(inputSeq)
    if L % 3 != 0:
        raise ValueError("length of input sequence must be a multiple of 3")
    out = np.empty((4, L), dtype=object)
    for k in range(0, L, 3):
        codon = inputSeq[k:k + 3]
        j = _CT_COL_IDX.get(codon)
        if j is None:
            raise ValueError(f"Unrecognized codon: {codon}")
        col = [CODON_TABLE_ARR[r, j] for r in range(12)]
        for pos in range(3):
            for nuc in range(4):
                out[nuc, k + pos] = col[pos * 4 + nuc]
    return out


def sampleMut(sim_leng, targeting, positions, rng):
    """Sample a (mutation, position) pair weighted by targeting."""
    flat = np.asarray(targeting).flatten(order="F")   # column-major
    total = flat.sum()
    if total <= 0:
        raise ValueError("No valid positions to mutate.")
    probs = flat / total
    pos = 0
    while pos in positions:
        mut = rng.choice(np.arange(1, 4 * sim_leng + 1), p=probs)
        pos = int(np.ceil(mut / 4))
    return {"mut": int(mut), "pos": pos}


def shmulateSeq(sequence, numMutations, targetingModel=None, start=1,
                end=None, frequency=False, rng=None):
    """Introduce random SHM mutations into a single sequence.

    Faithful port of shazam's ``shmulateSeq``.
    """
    if targetingModel is None:
        targetingModel = HH_S5F
    if rng is None:
        rng = np.random.default_rng()
    sequence = str(sequence)
    seq_len = len(sequence)
    if end is None:
        end = seq_len

    if not frequency:
        if abs(numMutations - round(numMutations)) > 1e-8:
            raise ValueError("numMutations must be a whole number.")
        numMutations = int(round(numMutations))
    else:
        if not (0 <= numMutations <= 1):
            raise ValueError("numMutations must be in [0, 1] for frequency.")

    if start < 1 or end > seq_len:
        raise ValueError("start must be >=1 and end <= sequence length.")
    head = sequence[:start - 1]
    tail = sequence[end:]
    sequence = sequence[start - 1:end]

    # trim to last codon
    L = len(sequence)
    cp = getCodonPos(L)
    if cp[2] > L:
        sim_seq = sequence[:cp[0] - 1]
        tail = sequence[cp[0] - 1:] + tail
    else:
        sim_seq = sequence
    sim_leng = len(sim_seq)
    if sim_leng % 3 != 0:
        raise ValueError("simulation sequence length not a multiple of 3")

    if frequency:
        numMutations = int(rng.binomial(sim_leng, numMutations))
    if numMutations > sim_leng:
        raise ValueError("numMutations larger than sequence length.")

    mutation_types = computeMutationTypes(sim_seq)
    targeting = calculateTargeting(
        germlineSeq=sim_seq, targetingModel=targetingModel).values
    targeting = targeting[:4]
    targeting = np.where(np.isnan(targeting), 0.0, targeting)
    targeting[mutation_types == "stop"] = 0.0

    sim_seq = list(sim_seq)
    # R: positions <- numeric(numMutations) -- fixed-size zero vector
    positions = [0] * numMutations
    total = 0
    while total < numMutations:
        mp = sampleMut(sim_leng, targeting, positions, rng)
        positions[total] = mp["pos"]
        total += 1
        # mut_nuc index: 4 - (4*pos - mut)
        mut_nuc = 4 - (4 * mp["pos"] - mp["mut"])
        sim_seq[mp["pos"] - 1] = _NUC[mut_nuc - 1]
        sim_str = "".join(sim_seq)
        lower = max(mp["pos"] - 4, 1)
        upper = min(mp["pos"] + 4, sim_leng)
        new_t = calculateTargeting(
            germlineSeq=sim_str[lower - 1:upper],
            targetingModel=targetingModel).values[:4]
        new_t = np.where(np.isnan(new_t), 0.0, new_t)
        targeting[:, lower - 1:upper] = new_t
        clow = getCodonPos(lower)[0]
        cup = getCodonPos(upper)[2]
        mt = computeMutationTypes(sim_str[clow - 1:cup])
        mutation_types[:, clow - 1:cup] = mt
        stop_mask = mt == "stop"
        if stop_mask.any():
            block = targeting[:, clow - 1:cup]
            block[stop_mask] = 0.0
            targeting[:, clow - 1:cup] = block

    return head + "".join(sim_seq) + tail


def shmulateTree(sequence, graph, targetingModel=None, field=None,
                 exclude=None, junctionWeight=None, start=1, end=None,
                 rng=None):
    """Simulate sequences along a lineage tree.

    ``graph`` must expose ``adjacency`` (dict node -> dict child -> weight),
    a ``mrca`` node name, and optionally ``vertex_attr`` (dict).
    Faithful port of shazam's ``shmulateTree``.
    """
    if targetingModel is None:
        targetingModel = HH_S5F
    if rng is None:
        rng = np.random.default_rng()

    adj = {p: dict(ch) for p, ch in graph["adjacency"].items()}
    mrca = graph["mrca"]
    nodes = graph.get("nodes", list(adj.keys()))
    skip = set()
    if field is not None and "vertex_attr" in graph:
        va = graph["vertex_attr"].get(field, {})
        exclude_set = (set(exclude) if isinstance(exclude, (list, set, tuple))
                       else {exclude})
        skip = {n for n, v in va.items() if v in exclude_set}

    seqs = {mrca: sequence}
    dists = {mrca: 0}

    if junctionWeight is not None and mrca in adj:
        for ch in adj[mrca]:
            adj[mrca][ch] = round(adj[mrca][ch] * (1 + junctionWeight))

    parents = [mrca]
    while parents:
        new_parents = []
        for p in parents:
            for ch, w in adj.get(p, {}).items():
                if w <= 0:
                    continue
                new_parents.append(ch)
                seqs[ch] = shmulateSeq(
                    seqs[p], numMutations=w, targetingModel=targetingModel,
                    start=start, end=end, rng=rng)
                dists[ch] = w
        parents = new_parents

    rows = []
    for n in nodes:
        if n == "Germline" or n == mrca and mrca == "Germline":
            continue
        if n in skip or n not in seqs:
            continue
        rows.append({"name": n, "sequence": seqs[n], "distance": dists[n]})
    import pandas as pd
    return pd.DataFrame(rows)
