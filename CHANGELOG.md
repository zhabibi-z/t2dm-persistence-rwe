# Changelog

All notable changes to this project are documented here.

## [1.0.0] — 2026-05-12

### Added
- **ETL**: `synthea_to_omop.py` — Synthea CSV → OMOP CDM v5.4 (DuckDB)
- **Cohort build**: `build_cohort.py` — 3 mutually exclusive new-user cohorts (n=5,000)
- **PS matching**: `cohort_matching.R` — 1:5 nearest-neighbour PS matching (MatchIt), love plots (cobalt)
- **TTD analysis**: `run_ttd.py` — Time-to-discontinuation with 90-day grace period (Lim 2025)
- **TTC analysis**: `run_ttc.py` — Per-comorbidity Kaplan-Meier
- **Cox TTD**: `run_cox.py` — Cox proportional hazards for TTD
- **Time-varying Cox**: `run_cox_timevarying.py` — Comorbidity 0→1 transitions as time-varying covariates
- **TTC Cox**: `run_cox_ttc.py` — Cox with dummy comorbidity interactions
- **Correlations**: `run_correlations.py` — Pearson correlation comorbidity × TTD + heatmap
- **Stratified KM**: `run_km_stratified.py` — Per-comorbidity KM grid (codx=0 vs codx=1)
- **R survival**: `survival_analysis.R` — survminer KM plots, Schoenfeld residuals, forest plot
- **Hypothesis tests**: `hypothesis_tests.R` — Shapiro-Wilk, Kruskal-Wallis, Dunn BH-FDR, Mann-Whitney U
- **ML module**: `ml/train.py` — XGBoost 5-fold CV (AUROC=0.961), SHAP explainability, UMAP phenotyping
- **Knowledge graph**: `graph/build_graph.py` — NetworkX DiGraph (19 nodes, 27 edges), Cypher export
- **Chatbot**: `chatbot/chatbot.py` — LangChain + Claude API + FAISS RAG + DuckDB SQL
- **Jupyter notebooks**: 5 notebooks in `analysis/notebooks/` (executed via nbconvert)
- **Streamlit dashboard**: `streamlit_app/app.py` — 6-tab interactive dashboard
- **R packages installer**: `scripts/install_r_packages.R`
- **Bootstrap script**: `scripts/bootstrap.sh` — full pipeline runner

### Results (synthetic data)
- Metformin: n=2,973, median TTD=407 days, 92.6% discontinued
- GLP-1 RA: n=1,056, median TTD=256 days, 95.5% discontinued
- SGLT-2i: n=971, median TTD=315 days, 93.7% discontinued
- XGBoost AUROC: 0.961 ± 0.008 (5-fold CV)
