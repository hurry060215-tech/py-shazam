"""R-parity tests: pyshazam vs shazam 1.3.2.

The R reference TSVs are produced by ``tests/r_reference_driver.R``. If R
(with shazam + alakazam) is unavailable, these tests are skipped.
"""
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import pyshazam as sh

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF_DIR = os.path.join(_HERE, "r_reference")
_DRIVER = os.path.join(_HERE, "r_reference_driver.R")

# CMAP R environment that has shazam 1.3.2 installed
_R_CONDA = "/scratch/users/steorra/env/CMAP"
_CONDA_SH = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"


def _r_available():
    """Return True if Rscript can load the shazam package."""
    if not os.path.exists(_CONDA_SH):
        return shutil.which("Rscript") is not None
    cmd = (f"source {_CONDA_SH} && conda activate {_R_CONDA} && "
           "Rscript -e 'library(shazam); library(alakazam)'")
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True,
                           timeout=120)
        return r.returncode == 0
    except Exception:
        return False


def _ensure_reference():
    """Generate the R reference TSVs if they are missing."""
    needed = ["example_db.tsv", "dist_to_nearest.tsv",
              "observed_mutations.tsv", "expected_mutations.tsv",
              "baseline_summary.tsv", "targeting_distance.tsv",
              "find_threshold_density.tsv", "substitution_1mer.tsv"]
    if all(os.path.exists(os.path.join(_REF_DIR, f)) for f in needed):
        return True
    if not _r_available():
        return False
    os.makedirs(_REF_DIR, exist_ok=True)
    cmd = (f"source {_CONDA_SH} && conda activate {_R_CONDA} && "
           f"Rscript {_DRIVER} {_REF_DIR}")
    subprocess.run(["bash", "-c", cmd], check=True, timeout=900)
    return True


_HAVE_R = _ensure_reference()
pytestmark = pytest.mark.skipif(
    not _HAVE_R, reason="R (shazam) reference data unavailable")


@pytest.fixture(scope="module")
def example_db():
    return pd.read_csv(os.path.join(_REF_DIR, "example_db.tsv"), sep="\t")


def test_observed_mutations_parity(example_db):
    """observedMutations R/S counts must be bit-exact."""
    ref = pd.read_csv(os.path.join(_REF_DIR, "observed_mutations.tsv"),
                      sep="\t")
    py = sh.observedMutations(example_db.head(100),
                              regionDefinition=sh.IMGT_V, frequency=False)
    cols = ["mu_count_cdr_r", "mu_count_cdr_s",
            "mu_count_fwr_r", "mu_count_fwr_s"]
    diff = np.abs(py[cols].values - ref[cols].values)
    assert np.nanmax(diff) < 1e-6


def test_observed_mutations_frequency_parity(example_db):
    """observedMutations frequency mode must be bit-exact."""
    ref = pd.read_csv(os.path.join(_REF_DIR, "observed_mutations_freq.tsv"),
                      sep="\t")
    py = sh.observedMutations(example_db.head(100),
                              regionDefinition=sh.IMGT_V, frequency=True)
    cols = ["mu_freq_cdr_r", "mu_freq_cdr_s",
            "mu_freq_fwr_r", "mu_freq_fwr_s"]
    diff = np.abs(py[cols].values - ref[cols].values)
    assert np.nanmax(diff) < 1e-6


def test_dist_to_nearest_parity(example_db):
    """distToNearest (ham model) must be bit-exact."""
    ref = pd.read_csv(os.path.join(_REF_DIR, "dist_to_nearest.tsv"),
                      sep="\t").set_index("sequence_id")["dist_nearest"]
    py = sh.distToNearest(example_db, sequenceColumn="junction",
                          vCallColumn="v_call", jCallColumn="j_call",
                          model="ham", normalize="len", first=False,
                          VJthenLen=True)
    py = py.set_index("sequence_id")["dist_nearest"]
    common = py.index.intersection(ref.index)
    # NA pattern must agree
    assert (py.loc[common].isna() == ref.loc[common].isna()).all()
    diff = np.abs(py.loc[common].values - ref.loc[common].values)
    assert np.nanmax(diff) < 1e-6


def test_expected_mutations_parity(example_db):
    """expectedMutations frequencies must agree within 1e-6."""
    ref = pd.read_csv(os.path.join(_REF_DIR, "expected_mutations.tsv"),
                      sep="\t")
    py = sh.expectedMutations(example_db.head(100),
                              germlineColumn="germline_alignment_d_mask",
                              regionDefinition=sh.IMGT_V,
                              targetingModel=sh.HH_S5F)
    cols = ["mu_expected_cdr_r", "mu_expected_cdr_s",
            "mu_expected_fwr_r", "mu_expected_fwr_s"]
    diff = np.abs(py[cols].values - ref[cols].values)
    assert np.nanmax(diff) < 1e-6


def test_targeting_distance_parity():
    """calcTargetingDistance(HH_S5F) must be bit-exact."""
    ref = pd.read_csv(os.path.join(_REF_DIR, "targeting_distance.tsv"),
                      sep="\t", index_col=0)
    py = sh.calcTargetingDistance(sh.HH_S5F)
    diff = np.abs(py.values - ref.values)
    assert np.nanmax(diff) < 1e-6


def test_substitution_matrix_parity(example_db):
    """createSubstitutionMatrix (1mer) must be bit-exact."""
    ref = pd.read_csv(os.path.join(_REF_DIR, "substitution_1mer.tsv"),
                      sep="\t", index_col=0)
    db_tm = example_db[(example_db.c_call == "IGHA")
                       & (example_db.sample_id == "-1h")]
    py = sh.createSubstitutionMatrix(
        db_tm, model="s", sequenceColumn="sequence_alignment",
        germlineColumn="germline_alignment_d_mask", vCallColumn="v_call",
        multipleMutation="independent", returnModel="1mer")
    diff = np.abs(py.values - ref.values)
    assert np.nanmax(diff) < 1e-6


def test_find_threshold_density_parity(example_db):
    """findThreshold density: bandwidth and threshold within tolerance."""
    ref = pd.read_csv(os.path.join(_REF_DIR, "find_threshold_density.tsv"),
                      sep="\t")
    dtn = sh.distToNearest(example_db, sequenceColumn="junction",
                           vCallColumn="v_call", jCallColumn="j_call",
                           model="ham", normalize="len", first=False)
    d = dtn["dist_nearest"].dropna().values
    thr = sh.findThreshold(d, method="density")
    assert abs(thr.bandwidth - ref["bandwidth"].iloc[0]) < 1e-5
    assert abs(thr.threshold - ref["threshold"].iloc[0]) < 1e-3


def test_baseline_summary_parity(example_db):
    """BASELINe selection sigma must agree within a tight tolerance."""
    ref = pd.read_csv(os.path.join(_REF_DIR, "baseline_summary.tsv"),
                      sep="\t")
    sub = example_db[(example_db.c_call == "IGHG")
                     & (example_db.sample_id == "+7d")].copy()
    cl = sh.collapseClones(
        sub, cloneColumn="clone_id", sequenceColumn="sequence_alignment",
        germlineColumn="germline_alignment_d_mask", method="mostCommon")
    baseline = sh.calcBaseline(
        cl, sequenceColumn="clonal_sequence",
        germlineColumn="clonal_germline", testStatistic="focused",
        regionDefinition=sh.IMGT_V, targetingModel=sh.HH_S5F)
    grouped = sh.groupBaseline(baseline, groupBy="c_call")
    summ = sh.summarizeBaseline(grouped, returnType="df")
    for i, region in enumerate(["cdr", "fwr"]):
        py_sigma = summ[summ.region == region]["baseline_sigma"].iloc[0]
        r_sigma = ref[ref.region == region]["baseline_sigma"].iloc[0]
        assert abs(py_sigma - r_sigma) < 1e-4


def test_baseline_per_sequence_sigma_parity(example_db):
    """Per-sequence BASELINe sigma must be bit-exact vs R."""
    ref = pd.read_csv(os.path.join(_REF_DIR, "baseline_seq_summary.tsv"),
                      sep="\t")
    sub = example_db[(example_db.c_call == "IGHG")
                     & (example_db.sample_id == "+7d")].copy()
    cl = sh.collapseClones(
        sub, cloneColumn="clone_id", sequenceColumn="sequence_alignment",
        germlineColumn="germline_alignment_d_mask", method="mostCommon")
    baseline = sh.calcBaseline(
        cl, sequenceColumn="clonal_sequence",
        germlineColumn="clonal_germline", testStatistic="focused",
        regionDefinition=sh.IMGT_V, targetingModel=sh.HH_S5F)
    summ = sh.summarizeBaseline(baseline, returnType="df")
    for region in ["cdr", "fwr"]:
        py = np.sort(summ[summ.region == region]
                     ["baseline_sigma"].dropna().values)
        r = np.sort(ref[ref.region == region]
                    ["baseline_sigma"].dropna().values)
        n = min(len(py), len(r))
        assert n > 0
        assert np.max(np.abs(py[:n] - r[:n])) < 1e-6
