#!/usr/bin/env Rscript
# R reference driver for py-shazam parity tests.
# Loads shazam's ExampleDb, runs distToNearest, findThreshold,
# observedMutations, calcBaseline -> groupBaseline -> summarizeBaseline,
# createTargetingModel, and writes numeric results to TSVs.

suppressPackageStartupMessages({
    library(shazam)
    library(alakazam)
})

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else tempdir()
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

set.seed(42)

# ---- Load example data ----
data(ExampleDb, package = "alakazam")
db <- ExampleDb

# Export the ExampleDb itself (for the Python side) -- a small subset of cols
keep_cols <- c("sequence_id", "sequence_alignment", "germline_alignment_d_mask",
               "v_call", "j_call", "c_call", "junction", "junction_length",
               "clone_id", "sample_id", "locus")
write.table(db[, keep_cols], file.path(outdir, "example_db.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# ---- distToNearest ----
dtn <- distToNearest(db, sequenceColumn = "junction", vCallColumn = "v_call",
                     jCallColumn = "j_call", model = "ham", normalize = "len",
                     first = FALSE, VJthenLen = TRUE, nproc = 1)
write.table(data.frame(sequence_id = dtn$sequence_id,
                       dist_nearest = dtn$dist_nearest),
            file.path(outdir, "dist_to_nearest.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# ---- findThreshold (density) ----
dist_vec <- dtn$dist_nearest[!is.na(dtn$dist_nearest)]
thr_dens <- findThreshold(dist_vec, method = "density")
write.table(data.frame(method = "density",
                       threshold = thr_dens@threshold,
                       bandwidth = thr_dens@bandwidth),
            file.path(outdir, "find_threshold_density.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# ---- observedMutations (R/S by region, IMGT_V) ----
om <- observedMutations(db[1:100, ], sequenceColumn = "sequence_alignment",
                        germlineColumn = "germline_alignment_d_mask",
                        regionDefinition = IMGT_V, frequency = FALSE)
om_cols <- c("sequence_id", "mu_count_cdr_r", "mu_count_cdr_s",
             "mu_count_fwr_r", "mu_count_fwr_s")
write.table(om[, om_cols], file.path(outdir, "observed_mutations.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# observedMutations frequency
omf <- observedMutations(db[1:100, ], sequenceColumn = "sequence_alignment",
                         germlineColumn = "germline_alignment_d_mask",
                         regionDefinition = IMGT_V, frequency = TRUE)
omf_cols <- c("sequence_id", "mu_freq_cdr_r", "mu_freq_cdr_s",
              "mu_freq_fwr_r", "mu_freq_fwr_s")
write.table(omf[, omf_cols], file.path(outdir, "observed_mutations_freq.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# ---- expectedMutations (IMGT_V) ----
em <- expectedMutations(db[1:100, ], sequenceColumn = "sequence_alignment",
                        germlineColumn = "germline_alignment_d_mask",
                        regionDefinition = IMGT_V, targetingModel = HH_S5F)
em_cols <- c("sequence_id", "mu_expected_cdr_r", "mu_expected_cdr_s",
             "mu_expected_fwr_r", "mu_expected_fwr_s")
write.table(em[, em_cols], file.path(outdir, "expected_mutations.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# ---- BASELINe: calcBaseline -> groupBaseline -> summarizeBaseline ----
db_sub <- subset(ExampleDb, c_call == "IGHG" & sample_id == "+7d")
db_clones <- collapseClones(db_sub, cloneColumn = "clone_id",
                            sequenceColumn = "sequence_alignment",
                            germlineColumn = "germline_alignment_d_mask",
                            method = "mostCommon")
baseline <- calcBaseline(db_clones,
                         sequenceColumn = "clonal_sequence",
                         germlineColumn = "clonal_germline",
                         testStatistic = "focused",
                         regionDefinition = IMGT_V,
                         targetingModel = HH_S5F, nproc = 1)
grouped <- groupBaseline(baseline, groupBy = "c_call")
summ <- summarizeBaseline(grouped, returnType = "df")
write.table(summ, file.path(outdir, "baseline_summary.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# per-sequence baseline sigma (before grouping)
seq_summ <- summarizeBaseline(baseline, returnType = "df")
write.table(seq_summ[, c("region", "baseline_sigma",
                         "baseline_ci_lower", "baseline_ci_upper",
                         "baseline_ci_pvalue")],
            file.path(outdir, "baseline_seq_summary.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# ---- calcTargetingDistance reference ----
td <- calcTargetingDistance(HH_S5F)
write.table(as.data.frame(td), file.path(outdir, "targeting_distance.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

# ---- createTargetingModel (small subset) ----
db_tm <- subset(ExampleDb, c_call == "IGHA" & sample_id == "-1h")
sub_mat <- createSubstitutionMatrix(db_tm, model = "s",
                                    sequenceColumn = "sequence_alignment",
                                    germlineColumn = "germline_alignment_d_mask",
                                    vCallColumn = "v_call",
                                    multipleMutation = "independent",
                                    returnModel = "1mer")
write.table(as.data.frame(sub_mat),
            file.path(outdir, "substitution_1mer.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

cat("R reference driver done. Outputs in", outdir, "\n")
