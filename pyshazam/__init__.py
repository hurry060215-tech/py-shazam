"""pyshazam -- pure-Python port of the R/CRAN package *shazam*.

shazam (Gupta, Vander Heiden, et al. 2015; part of the Immcantation
framework) is a computational toolkit for analyzing somatic hypermutation
(SHM) in immunoglobulin sequences. ``pyshazam`` is a faithful Python
re-implementation of its computational core -- no R, no rpy2.

Covered functionality
---------------------
* Distance-to-nearest + clonal threshold: ``distToNearest``,
  ``findThreshold`` (density and gmm methods), ``calcTargetingDistance``.
* SHM targeting models: ``createSubstitutionMatrix``,
  ``createMutabilityMatrix``, ``createTargetingMatrix``,
  ``createTargetingModel``, ``extendSubstitutionMatrix``,
  ``extendMutabilityMatrix``, plus the bundled HH_S5F / HH_S1F / MK_RS5NF /
  MK_RS1NF / HKL_S5F / HKL_S1F / U5N models.
* Mutation analysis: ``observedMutations`` / ``calcObservedMutations``,
  ``expectedMutations`` / ``calcExpectedMutations``, ``setRegionBoundaries``,
  ``slideWindowSeq`` / ``slideWindowDb`` / ``slideWindowTune``.
* BASELINe selection analysis: ``calcBaseline``, ``groupBaseline``,
  ``summarizeBaseline``, ``testBaseline``, ``createBaseline``,
  ``editBaseline``.
* SHM simulation: ``shmulateSeq``, ``shmulateTree``.
* Consensus / collapsing: ``collapseClones``, ``consensusSequence``.
* Plotting (matplotlib): distance / threshold, mutability heatmap,
  BASELINe density and summary, mutation frequency.

Faithful to shazam 1.3.2 (Kleinstein lab, Yale; AGPL-3).
"""

__version__ = "0.1.0"

# ---- Core: classes, constants, region/mutation definitions ----
from .core import (
    RegionDefinition, MutationDefinition,
    createRegionDefinition, makeNullRegionDefinition,
    createMutationDefinition, computeCodonTable, allCodonMuts,
    getCodonNumb, getCodonNucs, getContextInCodon, getCodonPos,
    mutationType, checkAmbiguousExist, convertNumbering,
    nucs2IUPAC, IUPAC2nucs,
    NUCLEOTIDES, NUCLEOTIDES_AMBIGUOUS, AMINO_ACIDS, REGION_PALETTE,
    IMGT_V, IMGT_V_BY_CODONS, IMGT_V_BY_REGIONS, IMGT_V_BY_SEGMENTS,
    IMGT_VDJ, IMGT_VDJ_BY_REGIONS, IMGT_SCHEMES,
    CHARGE_MUTATIONS, HYDROPATHY_MUTATIONS, POLARITY_MUTATIONS,
    VOLUME_MUTATIONS, MUTATION_SCHEMES,
)

# ---- Regions ----
from .regions import setRegionBoundaries

# ---- Mutation profiling ----
from .mutation import (
    calcObservedMutations, observedMutations, binMutationsByRegion,
    countNonNByRegion, slideWindowSeq, slideWindowSeqHelper,
    slideWindowDb, slideWindowTune,
)
from .expected import (
    calcExpectedMutations, expectedMutations, calculateTargeting,
    calculateMutationalPaths,
)

# ---- Targeting models ----
from .targeting import (
    TargetingModel, MutabilityModel,
    createSubstitutionMatrix, createMutabilityMatrix, createTargetingMatrix,
    createTargetingModel, extendSubstitutionMatrix, extendMutabilityMatrix,
    calcTargetingDistance, calculateMutability, rescaleMutability,
    symmetrize, writeTargetingDistance, getFamily,
    minNumMutationsTune, minNumSeqMutationsTune,
    HH_S5F, HH_S1F, MK_RS5NF, MK_RS1NF, HKL_S5F, HKL_S1F, U5N,
)

# ---- Distance to nearest / threshold ----
from .distance import (
    distToNearest, findThreshold, smoothValley, gmmFit, nearestDist,
    pairwiseDist, pairwise5MerDist, calcTargetingDistance as _ctd,
    getDNAMatrix, getAAMatrix, DensityThreshold, GmmThreshold,
    HH_S5F_Distance, HH_S1F_Distance, MK_RS5NF_Distance, MK_RS1NF_Distance,
    HS1F_Compat, M1N_Compat,
)

# ---- BASELINe selection ----
from .baseline import (
    Baseline, createBaseline, editBaseline, calcBaseline, groupBaseline,
    summarizeBaseline, testBaseline, calcBaselineBinomialPdf,
    baselineSigma, baselineCI, baselinePValue,
)

# ---- SHM simulation ----
from .shmulate import shmulateSeq, shmulateTree, computeMutationTypes

# ---- Consensus / collapsing ----
from .consensus import consensusSequence, collapseClones

# ---- Plotting ----
from .plotting import (
    plotDistThreshold, plotDensityThreshold, plotGmmThreshold,
    plotMutability, plotMutabilityHeatmap, plotBaselineDensity,
    plotBaselineSummary, plotMutationFrequency,
)

__all__ = [
    # core / definitions
    "RegionDefinition", "MutationDefinition", "createRegionDefinition",
    "makeNullRegionDefinition", "createMutationDefinition",
    "computeCodonTable", "allCodonMuts", "getCodonNumb", "getCodonNucs",
    "getContextInCodon", "getCodonPos", "mutationType",
    "checkAmbiguousExist", "convertNumbering", "nucs2IUPAC", "IUPAC2nucs",
    "setRegionBoundaries",
    "NUCLEOTIDES", "NUCLEOTIDES_AMBIGUOUS", "AMINO_ACIDS", "REGION_PALETTE",
    "IMGT_V", "IMGT_V_BY_CODONS", "IMGT_V_BY_REGIONS",
    "IMGT_V_BY_SEGMENTS", "IMGT_VDJ", "IMGT_VDJ_BY_REGIONS", "IMGT_SCHEMES",
    "CHARGE_MUTATIONS", "HYDROPATHY_MUTATIONS", "POLARITY_MUTATIONS",
    "VOLUME_MUTATIONS", "MUTATION_SCHEMES",
    # mutation analysis
    "calcObservedMutations", "observedMutations", "binMutationsByRegion",
    "countNonNByRegion", "calcExpectedMutations", "expectedMutations",
    "calculateTargeting", "calculateMutationalPaths",
    "slideWindowSeq", "slideWindowSeqHelper", "slideWindowDb",
    "slideWindowTune",
    # targeting models
    "TargetingModel", "MutabilityModel", "createSubstitutionMatrix",
    "createMutabilityMatrix", "createTargetingMatrix",
    "createTargetingModel", "extendSubstitutionMatrix",
    "extendMutabilityMatrix", "calcTargetingDistance",
    "calculateMutability", "rescaleMutability", "symmetrize",
    "writeTargetingDistance", "getFamily", "minNumMutationsTune",
    "minNumSeqMutationsTune",
    "HH_S5F", "HH_S1F", "MK_RS5NF", "MK_RS1NF", "HKL_S5F", "HKL_S1F", "U5N",
    # distance / threshold
    "distToNearest", "findThreshold", "smoothValley", "gmmFit",
    "nearestDist", "pairwiseDist", "pairwise5MerDist", "getDNAMatrix",
    "getAAMatrix", "DensityThreshold", "GmmThreshold",
    "HH_S5F_Distance", "HH_S1F_Distance", "MK_RS5NF_Distance",
    "MK_RS1NF_Distance", "HS1F_Compat", "M1N_Compat",
    # baseline
    "Baseline", "createBaseline", "editBaseline", "calcBaseline",
    "groupBaseline", "summarizeBaseline", "testBaseline",
    "calcBaselineBinomialPdf", "baselineSigma", "baselineCI",
    "baselinePValue",
    # simulation
    "shmulateSeq", "shmulateTree", "computeMutationTypes",
    # consensus
    "consensusSequence", "collapseClones",
    # plotting
    "plotDistThreshold", "plotDensityThreshold", "plotGmmThreshold",
    "plotMutability", "plotMutabilityHeatmap", "plotBaselineDensity",
    "plotBaselineSummary", "plotMutationFrequency",
]
