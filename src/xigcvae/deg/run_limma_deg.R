# limma differential expression analysis for AD versus CN.
#
# Model:
#   expression ~ diagnosis + age + sex + APOE4
#
# Diagnosis coefficient:
#   AD versus CN
#
# Inputs are prepared by run_deg_analysis.py.

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 1) {
  stop("Usage: Rscript run_limma_deg.R <project_root>")
}

root <- args[1]
r_libs <- file.path(root, "R_libs")

if (dir.exists(r_libs)) {
  .libPaths(c(r_libs, .libPaths()))
}

if (!requireNamespace("limma", quietly = TRUE)) {
  stop(
    paste(
      "The R package 'limma' is not installed or not visible.",
      "Install limma into R_libs or load the correct R environment first.",
      sep = "\n"
    )
  )
}

library(limma)

out_dir <- file.path(root, "outputs", "DEG")

expr_file <- file.path(out_dir, "limma_input_expression_matrix.csv")
feature_file <- file.path(out_dir, "limma_input_feature_info.csv")
meta_file <- file.path(out_dir, "limma_input_sample_metadata.csv")

expr_df <- read.csv(expr_file, check.names = FALSE, stringsAsFactors = FALSE)
feature_df <- read.csv(feature_file, check.names = FALSE, stringsAsFactors = FALSE)
meta <- read.csv(meta_file, check.names = FALSE, stringsAsFactors = FALSE)

rownames(expr_df) <- expr_df$probe_id
expr_df$probe_id <- NULL

expr_mat <- as.matrix(expr_df)
storage.mode(expr_mat) <- "numeric"

rownames(meta) <- meta$ge_sample_column
meta <- meta[colnames(expr_mat), , drop = FALSE]

meta$diagnosis <- relevel(factor(meta$diagnosis), ref = "CN")
meta$sex <- factor(meta$sex)
meta$age <- as.numeric(meta$age)
meta$apoe4_allele_count <- as.numeric(meta$apoe4_allele_count)

complete_samples <- complete.cases(
  meta[, c("diagnosis", "age", "sex", "apoe4_allele_count")]
)

meta <- meta[complete_samples, , drop = FALSE]
expr_mat <- expr_mat[, rownames(meta), drop = FALSE]

design <- model.matrix(
  ~ diagnosis + age + sex + apoe4_allele_count,
  data = meta
)

fit <- lmFit(expr_mat, design)
fit <- eBayes(fit)

coef_name <- "diagnosisAD"

if (!(coef_name %in% colnames(design))) {
  stop(paste("Could not find coefficient:", coef_name))
}

tt <- topTable(
  fit,
  coef = coef_name,
  number = Inf,
  adjust.method = "BH",
  sort.by = "P"
)

tt$probe_id <- rownames(tt)

results <- merge(
  tt,
  feature_df,
  by = "probe_id",
  all.x = TRUE,
  sort = FALSE
)

results <- results[order(results$P.Value), ]

results$rank_by_raw_p_value <- seq_len(nrow(results))
results$selected_for_464_gene_panel <- results$rank_by_raw_p_value <= 464
results$direction_in_AD <- ifelse(results$logFC > 0, "Higher in AD", "Lower in AD")

results_out <- data.frame(
  `Gene symbol` = results$gene_symbol,
  `Probe ID / feature ID` = results$probe_id,
  `LocusLink` = results$locus_link,
  `limma logFC AD vs CN` = results$logFC,
  `Average expression` = results$AveExpr,
  `Moderated t statistic` = results$t,
  `Raw P value` = results$P.Value,
  `FDR-adjusted P value` = results$adj.P.Val,
  `B statistic` = results$B,
  `Rank by raw P value` = results$rank_by_raw_p_value,
  `Selected for 464-gene panel` = results$selected_for_464_gene_panel,
  `Direction in AD` = results$direction_in_AD,
  check.names = FALSE
)

write.csv(
  results_out,
  file.path(out_dir, "deg_results_all_genes.csv"),
  row.names = FALSE
)

write.csv(
  head(results_out, 464),
  file.path(out_dir, "top_464_gene_panel.csv"),
  row.names = FALSE
)

write.csv(
  meta,
  file.path(out_dir, "deg_sample_metadata.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame(design),
  file.path(out_dir, "deg_design_matrix.csv"),
  row.names = TRUE
)

cat("limma DEG completed.\n")
cat("Samples used:", nrow(meta), "\n")
cat("Design terms:", paste(colnames(design), collapse = ", "), "\n")
cat("Features tested:", nrow(results_out), "\n")
cat("Nominal P < 0.05:", sum(results_out$`Raw P value` < 0.05), "\n")
cat("FDR < 0.05:", sum(results_out$`FDR-adjusted P value` < 0.05), "\n")
