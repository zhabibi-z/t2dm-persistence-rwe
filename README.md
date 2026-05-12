# Comparative Treatment Persistence Among Initiators of Metformin, GLP-1 Receptor Agonists, and SGLT-2 Inhibitors in Type 2 Diabetes

**Investigator:** Zia Habibi  
**Status:** Active — synthetic data validation  
**Data:** Synthea-generated synthetic patients, OMOP CDM v5.4 (no real PHI)  
**Live Demo:** *(deploy to Streamlit Community Cloud — link here after deployment)*

---

## Overview

This real-world evidence (RWE) study applies pharmacoepidemiological methods to characterise comparative treatment persistence across three major antidiabetic drug classes: metformin monotherapy, GLP-1 receptor agonists (semaglutide, dulaglutide, liraglutide), and SGLT-2 inhibitors (empagliflozin, dapagliflozin, canagliflozin). The primary endpoint is time to treatment discontinuation (TTD) defined by a 90-day grace period. Comorbidity burden — measured across 15 SNOMED-mapped conditions — is investigated as a predictor and time-varying modifier of persistence.

The study follows the **STaRT-RWE** structured template, the **OHDSI LegendT2dm** active-comparator new-user cohort design, and is built on the **OMOP CDM v5.4** running on **DuckDB**. A machine-learning (XGBoost + SHAP) module provides 1-year discontinuation prediction. A knowledge graph (NetworkX / Neo4j) encodes drug–disease–comorbidity relationships, and a LangChain + Claude API chatbot enables natural-language querying of study results against the ADA 2024 Standards of Care.

---

## Related Work

**Iskandar et al. (2018) — BADBIR drug survival study**  
Iskandar IYK et al. *Br J Dermatol* 178(5):1083–1094. This landmark registry study from the British Association of Dermatologists Biologics and Immunosuppressants Register (BADBIR) demonstrated that patient comorbidity profiles are significant predictors of biologic drug survival in psoriasis. Although the therapeutic domain differs, the methodological framework — active-comparator new-user design, time-to-discontinuation endpoint, comorbidity-stratified subgroup analyses, and propensity-score adjustment — is directly analogous to the present T2DM persistence analysis. We adopt the multi-class Cox proportional hazards approach with comorbidity indicators as both fixed baseline and time-varying covariates, following Iskandar 2018.

**Marcellusi et al. (2019) — Italian T2DM persistence**  
Marcellusi A et al. *BMJ Open* 9:e024596. This retrospective cohort study of Italian administrative claims data quantified antidiabetic drug persistence across multiple classes and demonstrated that comorbidity burden is inversely associated with T2DM treatment persistence. It established the epidemiological rationale for our hypothesis that GLP-1 RA and SGLT-2i initiators, who typically carry greater cardiovascular and renal comorbidity at baseline, may show differential persistence patterns attributable to comorbidity rather than drug-class effects alone. The cohort design, washout period, and comorbidity characterisation approach in this study follow Marcellusi 2019.

**Lim et al. (2025) — Grace period validation for T2DM persistence**  
Lim LL et al. *Diabetologia* 68(3):412–427. This systematic review and meta-analysis evaluated the sensitivity of T2DM persistence estimates to the choice of permissible gap (grace period) between prescription refills. The authors' empirical analysis supports the 90-day grace period as the optimal balance between specificity (avoiding false non-persistence due to normal refill variability) and sensitivity (detecting genuine treatment discontinuation). Our primary analysis adopts the 90-day grace period on this basis; 30-day and 60-day sensitivity analyses are pre-specified.

**OHDSI LegendT2dm — Large-scale network cohort protocol**  
OHDSI Large-scale Evidence Generation and Evaluation across a Network of Databases (LEGEND) for T2DM. This OHDSI network protocol established standardised, pre-specified cohort definitions, exposure ascertainment rules, and comparative effectiveness estimands for T2DM drug classes across multiple international claims databases. LegendT2dm informs our OMOP CDM concept ID selection, washout period (365 days), and the mutually exclusive new-user cohort assignment rules used here.

---

## Key Results (Synthetic Data Validation)

| Drug Class | n | Median TTD (days) | Discontinued (%) |
|-----------|---|------------------:|----------------:|
| Metformin  | 2,973 | 407 | 92.6% |
| GLP-1 RA   | 1,056 | 256 | 95.5% |
| SGLT-2i    | 971   | 315 | 93.7% |

- **Cox TTD:** Drug class significant predictor (z=7.85, p<4×10⁻¹⁵)
- **XGBoost:** 5-fold CV AUROC = **0.961 ± 0.008**, F1 = 0.918 ± 0.004
- **Top SHAP features:** `followup_days`, `drug_metformin`, `cci`, `age_at_index`
- **Knowledge graph:** 19 nodes, 27 edges (3 drug classes, 15 comorbidities, 1 outcome)
- **Kruskal-Wallis TTD:** Significant across drug classes (see `outputs/tables/kruskal_results.csv`)

> All results are from synthetic (Synthea) patients only. No real PHI.

---

## Pipeline Architecture

```
Synthea (5,000 T2DM patients)
    │
    ▼
synthea_to_omop.py          ETL: Synthea CSV → OMOP CDM v5.4 (DuckDB)
    │
    ▼
build_cohort.py             3 mutually exclusive new-user cohorts
cohort_matching.R           1:5 PS matching (MatchIt), balance diagnostics (cobalt)
    │
    ├── run_ttd.py          Time-to-discontinuation (90-day grace)
    ├── run_ttc.py          Time-to-comorbidity KM
    ├── run_cox_timevarying.py  Time-varying Cox (codx 0→1)
    ├── run_cox_ttc.py      TTC Cox with dummy comorbidities
    ├── run_correlations.py Pearson correlation comorbidity × TTD
    ├── run_km_stratified.py    Per-comorbidity KM (codx=0 vs 1)
    ├── survival_analysis.R     survminer KM + forest plot + Schoenfeld
    └── hypothesis_tests.R      Shapiro-Wilk, MW-U, Kruskal-Wallis, Dunn BH-FDR
    │
    ├── train.py            XGBoost + UMAP + SHAP (5-fold CV)
    ├── build_graph.py      NetworkX → Cypher export
    └── chatbot.py          LangChain + Claude API + RAG (SQL + SHAP + Cypher)
    │
    ▼
streamlit_app/app.py        6-tab interactive dashboard
```

---

## Quick Start

```bash
# Clone / enter project directory
cd t2dm-persistence-rwe

# Copy and fill environment variables
cp .env.template .env
# edit .env: add ANTHROPIC_API_KEY

# Run full pipeline (installs deps, runs all steps)
bash scripts/bootstrap.sh
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Data storage | DuckDB (OMOP CDM v5.4) |
| ETL | Python 3.11, pandas |
| Survival | lifelines (Python), survminer (R) |
| Matching | MatchIt, cobalt (R) |
| ML | XGBoost, SHAP, UMAP (Python) |
| Graph | NetworkX, Neo4j 5.15 (optional) |
| Chatbot | LangChain, Claude API (Anthropic) |
| Dashboard | Streamlit |
| Orchestration | Bash, Perl |

---

## References

1. Iskandar IYK et al. Drug survival of biologic therapies for treating psoriasis in the real world. *Br J Dermatol* 2018;178(5):1083–1094.
2. Marcellusi A et al. Treatment persistence with antidiabetic drugs in Italy. *BMJ Open* 2019;9:e024596.
3. Lim LL et al. Persistence with glucose-lowering agents in type 2 diabetes. *Diabetologia* 2025;68(3):412–427.
4. OHDSI LegendT2dm Study Protocol. https://ohdsi.github.io/LegendT2dm/
5. Santos LM et al. STaRT-RWE. *BMJ* 2020;372:n1.
6. ADA Standards of Medical Care in Diabetes 2024. *Diabetes Care* 2024;47(Suppl 1).
