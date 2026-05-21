"""Smoke tests for pyshazam: imports, data loading, basic numerics."""
import numpy as np
import pandas as pd
import pytest

import pyshazam as sh


def test_import_and_all():
    assert isinstance(sh.__all__, list)
    assert len(sh.__all__) > 80
    for name in sh.__all__:
        assert hasattr(sh, name), f"missing export: {name}"


def test_bundled_targeting_models():
    for m in (sh.HH_S5F, sh.MK_RS5NF, sh.HKL_S5F, sh.U5N):
        assert isinstance(m, sh.TargetingModel)
        assert m.targeting.shape == (5, 3125)
        assert len(m.mutability) == 3125
    # 1-mer substitution matrices
    for m in (sh.HH_S1F, sh.MK_RS1NF, sh.HKL_S1F):
        assert m.shape == (4, 4)


def test_region_definitions():
    assert sh.IMGT_V.seqLength == 312
    assert sh.IMGT_V.regions == ["cdr", "fwr"]
    assert len(sh.IMGT_V.boundaries) == 312
    assert sh.IMGT_V_BY_REGIONS.seqLength == 312
    rd = sh.makeNullRegionDefinition(30)
    assert rd.seqLength == 30


def test_mutation_definitions():
    for md in (sh.CHARGE_MUTATIONS, sh.HYDROPATHY_MUTATIONS,
               sh.POLARITY_MUTATIONS, sh.VOLUME_MUTATIONS):
        assert isinstance(md, sh.MutationDefinition)
        assert len(md.classes) > 0


def test_codon_helpers():
    assert sh.getCodonPos(5) == [4, 5, 6]
    assert sh.getContextInCodon(5) == 2
    assert sh.getCodonNumb(5) == 2
    assert len(sh.allCodonMuts("ATG")) == 12


def test_mutation_type():
    # silent: TTT -> TTC (both Phe)
    assert sh.mutationType("TTT", "TTC")["s"] == 1
    # replacement: TTT -> TTA (Phe -> Leu)
    assert sh.mutationType("TTT", "TTA")["r"] == 1
    # stop: TTT -> TGA partial? TGG -> TGA Trp -> stop
    assert sh.mutationType("TGG", "TGA")["stop"] == 1


def test_calc_observed_mutations():
    in_seq = "TTTACGACA"
    gl_seq = "TTTACGTCA"
    res = sh.calcObservedMutations(in_seq, gl_seq)
    assert res["seq_r"] + res["seq_s"] == 1.0
    # no mutation
    res2 = sh.calcObservedMutations(gl_seq, gl_seq)
    assert all(np.isnan(v) for v in res2.values())


def test_calc_targeting_distance():
    td = sh.calcTargetingDistance(sh.HH_S5F)
    assert td.shape == (5, 3125)
    # 1-mer model
    td1 = sh.calcTargetingDistance(sh.HH_S1F)
    assert td1.shape == (7, 7)


def test_compute_codon_table():
    arr, rn, cn = sh.computeCodonTable()
    assert arr.shape == (12, 216)


def test_slide_window():
    # 4 R/S mutations (the two stop mutations are excluded) < 6 -> False
    assert sh.slideWindowSeq("TCGTCGAAAA", "AAAAAAAAAA", mutThresh=6,
                             windowSize=10) is False
    # 2 R/S mutations -> False
    assert sh.slideWindowSeq("TCAAAAAAAA", "AAAAAAAAAA", mutThresh=6,
                             windowSize=10) is False
    # 4 R/S mutations within a window of 6 -> True at mutThresh=4
    assert sh.slideWindowSeq("TCGTCGAAAA", "AAAAAAAAAA", mutThresh=4,
                             windowSize=6) is True


def test_shmulate_seq_deterministic_length():
    rng = np.random.default_rng(0)
    seq = "NGATCTGACGACACGGCCGTGTATTACTGTGCGAGAGATAGTTTA"
    out = sh.shmulateSeq(seq, numMutations=5, rng=rng)
    assert len(out) == len(seq)
    assert out != seq


def test_consensus_sequence():
    seqs = ["ACGTACGT", "ACGTACGT", "ACGTTCGT"]
    res = sh.consensusSequence(seqs, method="mostCommon")
    assert res["cons"] == "ACGTACGT"


def test_calc_baseline_binomial_pdf():
    pdf = sh.calcBaselineBinomialPdf(x=5, n=20, p=0.3)
    assert len(pdf) == 4001
    assert not np.any(np.isnan(pdf))
    # integrates to ~100 with default normalization
    assert abs(np.sum(pdf) - 100) < 1.0


def test_find_threshold_density_synthetic():
    rng = np.random.default_rng(1)
    d = np.concatenate([rng.normal(0.1, 0.03, 600),
                        rng.normal(0.5, 0.08, 600)])
    d = d[d > 0]
    thr = sh.findThreshold(d, method="density")
    assert isinstance(thr, sh.DensityThreshold)
    assert thr.threshold is not None
    assert 0.1 < thr.threshold < 0.5
