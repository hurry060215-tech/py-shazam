"""Extended region definitions (R/RegionsExtend.R: setRegionBoundaries)."""
from __future__ import annotations

import numpy as np

from .core import (RegionDefinition, makeNullRegionDefinition,
                   IMGT_V_BY_REGIONS)


def setRegionBoundaries(juncLength, sequenceImgt, regionDefinition=None):
    """Populate region boundaries for IMGT_VDJ / IMGT_VDJ_BY_REGIONS schemes.

    Faithful port of shazam's ``setRegionBoundaries``.
    """
    if regionDefinition is None:
        return makeNullRegionDefinition(len(sequenceImgt))
    if not isinstance(regionDefinition, RegionDefinition):
        raise ValueError("regionDefinition is not a valid RegionDefinition.")
    if regionDefinition.name not in ("IMGT_VDJ_BY_REGIONS", "IMGT_VDJ"):
        return regionDefinition

    seqLength = len(sequenceImgt)
    # mask of non-gap positions (1-based logic, 0-based array)
    helper = np.array([0 if ch in ("-", ".") else 1
                       for ch in sequenceImgt])
    helper[:309] = 0     # R: junction_length_helper[1:310 - 1] <- 0

    juncLength = int(juncLength) if juncLength is not None else 0
    if juncLength > 0:
        cs = np.cumsum(helper)
        idx = np.where(cs == juncLength)[0]
        if len(idx) == 0:
            junction_end = seqLength
            cdr3_end = junction_end
        else:
            junction_end = idx[0] + 1   # 1-based
            num_gaps = int(np.sum(
                helper[309:junction_end] == 0))
            juncLength = juncLength + num_gaps
            cdr3_end = 313 + juncLength - 6 - 1
    else:
        cdr3_end = 0

    base = list(IMGT_V_BY_REGIONS.boundaries)
    levels = list(IMGT_V_BY_REGIONS.boundaries_levels) + ["cdr3", "fwr4"]
    boundaries = list(base)
    # pad to seqLength
    while len(boundaries) < seqLength:
        boundaries.append(None)

    if cdr3_end > 312:
        for i in range(312, min(cdr3_end, seqLength)):
            boundaries[i] = "cdr3"
        if cdr3_end < seqLength:
            for i in range(cdr3_end, seqLength):
                boundaries[i] = "fwr4"

    if regionDefinition.name == "IMGT_VDJ":
        new_b = []
        for b in boundaries:
            if b is None:
                new_b.append(None)
            elif b.startswith("fwr"):
                new_b.append("fwr")
            elif b.startswith("cdr"):
                new_b.append("cdr")
            else:
                new_b.append(b)
        boundaries = new_b
        levels = ["fwr", "cdr"]

    return RegionDefinition(
        name=regionDefinition.name, description=regionDefinition.description,
        boundaries=boundaries, boundaries_levels=levels, seqLength=seqLength,
        regions=regionDefinition.regions, labels=regionDefinition.labels,
        citation=regionDefinition.citation)
