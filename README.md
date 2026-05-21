# py-shazam

Pure-Python port of the R/CRAN package
**[shazam](https://cran.r-project.org/package=shazam)** -- *Immunoglobulin
Somatic Hypermutation Analysis*, part of the
[Immcantation](https://immcantation.readthedocs.io) framework
(Gupta, Vander Heiden, et al., *Bioinformatics* 2015; Kleinstein Lab, Yale).

`pyshazam` is a standalone, dependency-light implementation of shazam's
computational core: distance-to-nearest neighbour, clonal-threshold
detection, SHM targeting models, observed/expected mutation profiling and
the BASELINe antigen-driven selection framework. **It does not require R.**

| | |
|---|---|
| PyPI / import name | `pyshazam` |
| License | AGPL-3 (same as upstream shazam) |
| Upstream | CRAN shazam 1.3.2 |

## Install

```bash
pip install pyshazam
```

Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`. The Hartigans'
dip-test p-value for the GMM threshold method is optional
(`pip install pyshazam[diptest]`).

## What is ported

`pyshazam` is a *faithful* re-implementation -- numerical parity with
shazam 1.3.2 is the design goal. All the shazam bundled data
(`HH_S5F`, `HH_S1F`, `MK_RS5NF`, `MK_RS1NF`, `HKL_S5F`, `HKL_S1F`,
`U5N`, the IMGT region schemes and the mutation-class schemes) ship inside
the package.

* **Distance-to-nearest + clonal threshold** -- `distToNearest`,
  `findThreshold` (`density` and `gmm` methods), `calcTargetingDistance`,
  `nearestDist`, `getDNAMatrix`, `getAAMatrix`.
* **SHM targeting models** -- `createSubstitutionMatrix`,
  `createMutabilityMatrix`, `createTargetingMatrix`, `createTargetingModel`,
  `extendSubstitutionMatrix`, `extendMutabilityMatrix`,
  `calculateMutability`, `symmetrize`.
* **Mutation analysis** -- `observedMutations` / `calcObservedMutations`,
  `expectedMutations` / `calcExpectedMutations`, `setRegionBoundaries`,
  `slideWindowSeq` / `slideWindowDb` / `slideWindowTune`.
* **BASELINe selection** -- `calcBaseline`, `groupBaseline`,
  `summarizeBaseline`, `testBaseline`, `createBaseline`, `editBaseline`.
* **SHM simulation** -- `shmulateSeq`, `shmulateTree`.
* **Consensus / collapsing** -- `collapseClones`, `consensusSequence`.
* **Plotting** (matplotlib) -- distance/threshold histograms, mutability
  heatmaps, BASELINe density and summary plots, mutation-frequency plots.

## Quick start

```python
import pandas as pd
import pyshazam as sh

# AIRR-format data frame of IMGT-aligned Ig sequences
db = pd.read_csv("example_db.tsv", sep="\t")

# --- Distance to nearest + clonal threshold ---
dtn = sh.distToNearest(db, sequenceColumn="junction",
                       vCallColumn="v_call", jCallColumn="j_call",
                       model="ham", normalize="len", first=False)
thr = sh.findThreshold(dtn["dist_nearest"].dropna().values, method="density")
print("clonal threshold:", thr.threshold)

# --- Observed mutations (R/S by CDR/FWR region) ---
db = sh.observedMutations(db, regionDefinition=sh.IMGT_V, frequency=False)

# --- BASELINe selection analysis ---
clones = sh.collapseClones(db, cloneColumn="clone_id",
                           sequenceColumn="sequence_alignment",
                           germlineColumn="germline_alignment_d_mask",
                           method="mostCommon")
baseline = sh.calcBaseline(clones, sequenceColumn="clonal_sequence",
                           germlineColumn="clonal_germline",
                           testStatistic="focused",
                           regionDefinition=sh.IMGT_V,
                           targetingModel=sh.HH_S5F)
grouped = sh.groupBaseline(baseline, groupBy="sample_id")
summary = sh.summarizeBaseline(grouped, returnType="df")
print(summary)

# --- Build a custom SHM targeting model ---
model = sh.createTargetingModel(db, model="s",
                                sequenceColumn="sequence_alignment",
                                germlineColumn="germline_alignment_d_mask",
                                vCallColumn="v_call")
```

## R-parity

py-shazam is validated against shazam 1.3.2. The deterministic functions
match R to machine precision:

| Function | Agreement vs shazam 1.3.2 |
|---|---|
| `observedMutations` (R/S counts and frequencies) | bit-exact (max abs diff 0) |
| `distToNearest` (Hamming model) | bit-exact (max abs diff 0) |
| `calcTargetingDistance` (HH_S5F) | bit-exact (max abs diff 0) |
| `createSubstitutionMatrix` (1-mer) | rel-diff < 1e-15 |
| `expectedMutations` | rel-diff < 1e-15 |
| `findThreshold` density bandwidth / threshold | bit-exact (< 1e-8) |
| BASELINe selection sigma (`summarizeBaseline`) | rel-diff < 1e-13 |
| `baselineCI` confidence intervals | rel-diff < 1e-8 |

`tests/test_r_parity.py` regenerates the R references from
`tests/r_reference_driver.R` and asserts these tolerances; it is skipped
automatically when R / shazam is unavailable.

## Citation

If you use py-shazam, please cite the original shazam package:

> Gupta NT, Vander Heiden JA, Uduman M, Gadala-Maria D, Yaari G,
> Kleinstein SH. *Change-O: a toolkit for analyzing large-scale B cell
> immunoglobulin repertoire sequencing data.* Bioinformatics 2015.

and, for the BASELINe selection methods and the SHM targeting models:

> Yaari G, Uduman M, Kleinstein SH. *Quantifying selection in
> high-throughput immunoglobulin sequencing data sets.* Nucleic Acids
> Research 2012; 40(17):e134.
>
> Yaari G, Vander Heiden JA, et al. *Models of somatic hypermutation
> targeting and substitution based on synonymous mutations from
> high-throughput immunoglobulin sequencing data.* Frontiers in
> Immunology 2013; 4:358.

## License

AGPL-3, the same license as the upstream shazam package. See `LICENSE`.
