"""Benchmark pyshazam on the shazam ExampleDb.

Times the core pipeline -- distToNearest, findThreshold,
observedMutations, expectedMutations, collapseClones and the BASELINe
selection workflow -- and prints a short report.
"""
import os
import time

import numpy as np
import pandas as pd

import pyshazam as sh

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.join(_HERE, "..", "tests", "r_reference", "example_db.tsv")


def _timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:<34s} {dt:8.3f} s")
    return out, dt


def main():
    if not os.path.exists(_REF):
        raise SystemExit(
            "ExampleDb not found -- run tests/r_reference_driver.R first.")
    db = pd.read_csv(_REF, sep="\t")
    print(f"shazam ExampleDb: {len(db)} sequences\n")
    print("Timing pyshazam core pipeline:")

    dtn, _ = _timed("distToNearest (ham)", lambda: sh.distToNearest(
        db, sequenceColumn="junction", vCallColumn="v_call",
        jCallColumn="j_call", model="ham", normalize="len", first=False))

    dist = dtn["dist_nearest"].dropna().values
    thr, _ = _timed("findThreshold (density)",
                    lambda: sh.findThreshold(dist, method="density"))

    _timed("observedMutations (IMGT_V, 200)", lambda: sh.observedMutations(
        db.head(200), regionDefinition=sh.IMGT_V, frequency=False))

    _timed("expectedMutations (IMGT_V, 200)", lambda: sh.expectedMutations(
        db.head(200), germlineColumn="germline_alignment_d_mask",
        regionDefinition=sh.IMGT_V, targetingModel=sh.HH_S5F))

    sub = db[(db.c_call == "IGHG") & (db.sample_id == "+7d")].copy()
    cl, _ = _timed("collapseClones (IGHG +7d)", lambda: sh.collapseClones(
        sub, cloneColumn="clone_id", sequenceColumn="sequence_alignment",
        germlineColumn="germline_alignment_d_mask", method="mostCommon"))

    baseline, _ = _timed("calcBaseline (focused)", lambda: sh.calcBaseline(
        cl, sequenceColumn="clonal_sequence",
        germlineColumn="clonal_germline", testStatistic="focused",
        regionDefinition=sh.IMGT_V, targetingModel=sh.HH_S5F))

    grouped, _ = _timed("groupBaseline",
                        lambda: sh.groupBaseline(baseline, groupBy="c_call"))
    summ, _ = _timed("summarizeBaseline",
                     lambda: sh.summarizeBaseline(grouped, returnType="df"))

    print("\nResults:")
    print(f"  clonal threshold (density): {thr.threshold:.4f}")
    print(f"  collapsed clones:           {len(cl)}")
    print("  BASELINe selection sigma:")
    for _, row in summ.iterrows():
        print(f"    {row['region']:<5s} sigma = {row['baseline_sigma']:+.4f}")


if __name__ == "__main__":
    main()
