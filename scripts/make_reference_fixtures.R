# Generate tests/fixtures/r_backbone.json from the R backbone package.
#
# The fixture pins the p-values (both tails) and the retained edges of
# R backbone (CRAN package "backbone", version 3.0 or later) on the shared
# reference datasets, so tests/test_reference_implementations.py can compare
# networkx-backbone against R without R installed.
#
# Usage (from the repository root):
#
#   python scripts/export_reference_inputs.py     # refresh the CSV inputs
#   Rscript scripts/make_reference_fixtures.R      # writes the JSON fixture
#
# Requirements: R >= 4.1, install.packages(c("backbone", "jsonlite")).
#
# Conventions in the JSON output:
# - Node indices are 0-based positions in the "nodes"/"agents" lists.
# - P-values are listed per edge (weighted models) or per agent pair i < j in
#   row-major order (bipartite models); null marks a p-value R leaves NA.
# - "upper" is the upper-tail p-value P(X >= observed) that the one-tailed
#   backbone thresholds; "lower" is the lower-tail value used by signed = TRUE.
# - Backbones are computed with R's defaults (missing_as_zero = FALSE) at
#   alpha = 0.05 and list the retained edges as [i, j, sign].

suppressPackageStartupMessages({
  library(backbone)
  library(jsonlite)
})

if (packageVersion("backbone") < "3.0.0") {
  stop("backbone >= 3.0.0 is required")
}

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
root <- if (length(script)) normalizePath(file.path(dirname(script), "..")) else getwd()
inputs <- file.path(root, "tests", "fixtures", "reference_inputs")
out_file <- file.path(root, "tests", "fixtures", "r_backbone.json")

ALPHA <- 0.05
MTC <- c("none", "bonferroni", "holm", "BH")
FDSM_TRIALS <- 10000
SEED <- 20261101

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

read_weighted <- function(path) {
  df <- utils::read.csv(path, colClasses = c("character", "character", "numeric"))
  nodes <- unique(c(df$source, df$target))
  A <- matrix(0, length(nodes), length(nodes), dimnames = list(nodes, nodes))
  A[cbind(df$source, df$target)] <- df$weight
  A[cbind(df$target, df$source)] <- df$weight
  list(A = A, nodes = nodes, edges = df)
}

read_incidence <- function(path) {
  df <- utils::read.csv(path, check.names = FALSE, colClasses = "character")
  I <- as.matrix(df[, -1])
  storage.mode(I) <- "numeric"
  dimnames(I) <- list(df[[1]], colnames(df)[-1])
  I
}

# Per-edge values of a matrix, in edge-list order (R drops the dimnames of
# the p-value matrices, so look edges up by position).
edge_values <- function(M, g) {
  as.vector(M[cbind(match(g$edges$source, g$nodes),
                    match(g$edges$target, g$nodes))])
}

# Upper-triangle (i < j) values in row-major order, matching
# numpy.triu_indices(n, 1).
pair_index <- function(n) {
  idx <- which(upper.tri(matrix(0, n, n)), arr.ind = TRUE)
  idx[order(idx[, 1], idx[, 2]), , drop = FALSE]
}

pair_values <- function(M) {
  as.vector(M[pair_index(nrow(M))])
}

# Evaluate quietly: fdsm prints a progress bar to stdout.
quietly <- function(expr) {
  out <- NULL
  utils::capture.output(out <- suppressMessages(expr))
  out
}

# Retained edges of a backbone matrix as [i, j, sign] with 0-based i < j.
retained <- function(bb) {
  bb <- as.matrix(bb)
  idx <- which(upper.tri(bb) & bb != 0, arr.ind = TRUE)
  idx <- idx[order(idx[, 1], idx[, 2]), , drop = FALSE]
  lapply(seq_len(nrow(idx)), function(r) {
    c(idx[r, 1] - 1, idx[r, 2] - 1, bb[idx[r, 1], idx[r, 2]])
  })
}

weighted_backbones <- function(A, model) {
  out <- list()
  for (signed in c(FALSE, TRUE)) {
    for (mtc in MTC) {
      key <- paste0(if (signed) "signed_" else "", mtc)
      bb <- backbone_from_weighted(A, model = model, alpha = ALPHA,
                                   signed = signed, mtc = mtc)
      out[[key]] <- retained(bb)
    }
  }
  out
}

projection_backbones <- function(I, model, mtcs = MTC, trials = NULL) {
  out <- list()
  for (signed in c(FALSE, TRUE)) {
    for (mtc in mtcs) {
      key <- paste0(if (signed) "signed_" else "", mtc)
      if (!is.null(trials)) set.seed(SEED)
      bb <- quietly(backbone_from_projection(
        I, model = model, alpha = ALPHA, signed = signed, mtc = mtc,
        trials = trials))
      out[[key]] <- retained(bb)
    }
  }
  out
}

# Both tails of a weighted model for every edge (signed = TRUE reports the
# upper tail unchanged and adds the lower tail).
weighted_tails <- function(g, model) {
  res <- backbone_from_weighted(g$A, model = model, alpha = ALPHA, signed = TRUE,
                                backbone_only = FALSE)
  list(upper = edge_values(res$pvalues$upper, g),
       lower = edge_values(res$pvalues$lower, g))
}

# Both tails of a projection model for every agent pair, including pairs
# that never co-occur (missing_as_zero = TRUE).
pair_tails <- function(I, model, trials = NULL) {
  if (!is.null(trials)) set.seed(SEED)
  res <- quietly(backbone_from_projection(
    I, model = model, alpha = ALPHA, signed = TRUE, missing_as_zero = TRUE,
    trials = trials, backbone_only = FALSE))
  list(upper = pair_values(res$pvalues$upper),
       lower = pair_values(res$pvalues$lower))
}

# ---------------------------------------------------------------------------
# Weighted models: Les Miserables (and karate)
# ---------------------------------------------------------------------------

weighted_section <- function(path) {
  g <- read_weighted(path)
  idx <- match(c(g$edges$source, g$edges$target), g$nodes) - 1
  m <- nrow(g$edges)
  sec <- list(
    nodes = g$nodes,
    edges = lapply(seq_len(m), function(r) c(idx[r], idx[m + r]))
  )
  for (model in c("disparity", "mlf", "lans")) {
    sec[[model]] <- c(weighted_tails(g, model),
                      list(backbones = weighted_backbones(g$A, model)))
  }
  sec
}

les <- weighted_section(file.path(inputs, "les_miserables.csv"))
karate <- weighted_section(file.path(inputs, "karate.csv"))

# Global threshold (the "global" model of backbone_from_weighted).
# stats::p.adjust applied to the Les Miserables MLF upper-tail p-values
# (many ties), the correction behind every `mtc` option.
les$p_adjust <- lapply(
  c(bonferroni = "bonferroni", holm = "holm", hochberg = "hochberg",
    BH = "BH", BY = "BY"),
  function(method) stats::p.adjust(les$mlf$upper, method = method)
)

les_A <- read_weighted(file.path(inputs, "les_miserables.csv"))$A
les$global <- list(
  backbones = list(
    threshold_2 = retained(backbone_from_weighted(les_A, model = "global",
                                                  parameter = 2)),
    signed_1_4 = retained(backbone_from_weighted(les_A, model = "global",
                                                 parameter = c(1, 4)))
  )
)

# ---------------------------------------------------------------------------
# Unweighted models: karate club (binary)
# ---------------------------------------------------------------------------

karate_U <- (read_weighted(file.path(inputs, "karate.csv"))$A > 0) * 1
karate$unweighted <- list()
for (model in c("lspar", "degree")) {
  for (s in c(0.3, 0.5, 0.7)) {
    key <- paste0(model, "_", s)
    karate$unweighted[[key]] <- retained(
      backbone_from_unweighted(karate_U, model = model, parameter = s))
  }
}

# lspar on the binary Les Miserables graph, where R's dense ranks and
# zero-Jaccard rule matter more than on the karate club (issue #59).
les_U <- (les_A > 0) * 1
les$unweighted <- list()
for (s in c(0.3, 0.5, 0.7)) {
  les$unweighted[[paste0("lspar_", s)]] <- retained(
    backbone_from_unweighted(les_U, model = "lspar", parameter = s))
}

# ---------------------------------------------------------------------------
# Bipartite projections: Davis Southern Women (women as agents)
# ---------------------------------------------------------------------------

I <- read_incidence(file.path(inputs, "davis_southern_women.csv"))
P <- tcrossprod(I)
davis <- list(
  agents = rownames(I),
  artifacts = colnames(I),
  pairs = lapply(seq_len(nrow(pair_index(nrow(I)))), function(r) {
    pair_index(nrow(I))[r, ] - 1
  }),
  cooccurrence = pair_values(P),
  bicm = unname(lapply(seq_len(nrow(I)), function(r) bicm(I)[r, ]))
)
for (model in c("sdsm", "fixedfill", "fixedrow", "fixedcol")) {
  davis[[model]] <- c(pair_tails(I, model),
                      list(backbones = projection_backbones(I, model)))
}
# SDSM with edge constraints (SDSM-EC): backbone codes a prohibited cell
# (an absent edge that cannot occur) as 10 and a required cell (a present
# edge that must occur) as 11.
EC_PROHIBITED <- list(c("Evelyn Jefferson", "E10"), c("Evelyn Jefferson", "E11"),
                      c("Laura Mandeville", "E12"), c("Brenda Rogers", "E13"))
EC_REQUIRED <- list(c("Evelyn Jefferson", "E1"), c("Laura Mandeville", "E2"),
                    c("Nora Fayette", "E11"))
I_ec <- I
for (cell in EC_PROHIBITED) {
  stopifnot(I_ec[cell[1], cell[2]] == 0)
  I_ec[cell[1], cell[2]] <- 10
}
for (cell in EC_REQUIRED) {
  stopifnot(I_ec[cell[1], cell[2]] == 1)
  I_ec[cell[1], cell[2]] <- 11
}
davis$sdsm_ec <- c(
  list(prohibited = EC_PROHIBITED, required = EC_REQUIRED),
  pair_tails(I_ec, "sdsm"),
  list(backbones = projection_backbones(I_ec, "sdsm"))
)

davis$fdsm <- c(
  list(trials = FDSM_TRIALS, seed = SEED),
  pair_tails(I, "fdsm", trials = FDSM_TRIALS),
  list(backbones = projection_backbones(I, "fdsm", mtcs = "none",
                                        trials = FDSM_TRIALS))
)

# backbone's fdsm() draws each sample with fastball() started from the
# observed matrix (rotated so rows are the smaller side) and only 5 trades
# per row.  On Davis that is 70 trades, too few to forget the observed
# matrix, so its p-values are biased upward.  This reference repeats the
# Monte Carlo test with FDSM_TRADES trades per row, using backbone's own
# fastball(), so the sampler is mixed.
FDSM_TRADES <- 100
fdsm_mixed <- function(I, trials, trades_per_row) {
  set.seed(SEED)
  P <- tcrossprod(I)
  M <- if (nrow(I) > ncol(I)) t(I) else I
  upper <- matrix(0, nrow(P), ncol(P))
  lower <- matrix(0, nrow(P), ncol(P))
  for (t in seq_len(trials)) {
    S <- fastball(M, trades = trades_per_row * nrow(M))
    Pstar <- if (nrow(I) > ncol(I)) crossprod(S) else tcrossprod(S)
    upper <- upper + (Pstar >= P)
    lower <- lower + (Pstar <= P)
  }
  list(trials = trials, trades_per_row = trades_per_row,
       upper = pair_values(upper / trials), lower = pair_values(lower / trials))
}
davis$fdsm_mixed <- fdsm_mixed(I, FDSM_TRIALS, FDSM_TRADES)

# Which pairs count as tests for `mtc` (issue #59).  With missing_as_zero =
# FALSE (R's default) pairs that never co-occur get no p-value, so they are
# not counted by p.adjust and never retained; with TRUE every pair is tested.
# At alpha = 0.05 no Davis backbone depends on this, so these backbones use
# larger alphas at which it does.
MAZ_ALPHAS <- c(0.06, 0.6, 0.8)
davis$missing_as_zero <- list(alphas = MAZ_ALPHAS)
for (model in c("fixedfill", "fixedrow")) {
  davis$missing_as_zero[[model]] <- list()
  for (maz in c(FALSE, TRUE)) {
    by_alpha <- list()
    for (a in MAZ_ALPHAS) {
      out <- list()
      for (signed in c(FALSE, TRUE)) {
        for (mtc in MTC) {
          key <- paste0(if (signed) "signed_" else "", mtc)
          out[[key]] <- retained(backbone_from_projection(
            I, model = model, alpha = a, signed = signed, mtc = mtc,
            missing_as_zero = maz))
        }
      }
      by_alpha[[as.character(a)]] <- out
    }
    davis$missing_as_zero[[model]][[tolower(as.character(maz))]] <- by_alpha
  }
}

# ---------------------------------------------------------------------------
# Write the fixture
# ---------------------------------------------------------------------------

fixture <- list(
  meta = list(
    generator = "scripts/make_reference_fixtures.R",
    R = paste(R.version$major, R.version$minor, sep = "."),
    backbone = as.character(packageVersion("backbone")),
    jsonlite = as.character(packageVersion("jsonlite")),
    rng = paste(RNGkind(), collapse = "/"),
    alpha = ALPHA,
    generated = format(Sys.Date())
  ),
  les_miserables = les,
  karate = karate,
  davis = davis
)

write_json(fixture, out_file, auto_unbox = TRUE, digits = NA, na = "null",
           pretty = FALSE)
message("wrote ", out_file)
