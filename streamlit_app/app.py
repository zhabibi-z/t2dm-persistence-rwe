"""
streamlit_app/app.py — T2DM Persistence RWE interactive dashboard.

6 tabs:
  1. Overview     — study background, citations, data flow diagram
  2. Cohort       — baseline characteristics, comorbidity prevalence, PS matching
  3. Survival     — KM curves, Cox HR tables, forest plot, stratified KM
  4. ML           — XGBoost CV performance, SHAP, UMAP
  5. Graph        — knowledge graph visualisation, Cypher queries
  6. Chatbot      — LangChain + Groq (Llama 3.3 70B) Q&A

Run: streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

st.set_page_config(
    page_title="T2DM Persistence RWE",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: str, default_cols: list[str] | None = None) -> pd.DataFrame | None:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None


def show_image(path: str, caption: str = "", width: int | None = None) -> None:
    p = Path(path)
    if p.exists():
        st.image(str(p), caption=caption, use_column_width=(width is None))
    else:
        st.info(f"Figure not yet generated: `{path}`\nRun `bash scripts/bootstrap.sh` to produce outputs.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("T2DM Persistence RWE")
    st.markdown("**Investigator:** Zia Habibi")
    st.markdown("**Data:** Synthea synthetic patients, OMOP CDM v5.4")
    st.markdown("**Status:** Synthetic data validation")
    st.divider()
    st.markdown("**Key References**")
    st.markdown("- Lim 2025 (90-day grace)")
    st.markdown("- Iskandar 2018 (Cox PH)")
    st.markdown("- Marcellusi 2019 (cohort design)")
    st.markdown("- OHDSI LegendT2dm")
    st.divider()
    st.caption("Synthetic data only — no real PHI")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Overview", "👥 Cohort", "📈 Survival", "🤖 ML", "🕸️ Graph", "💬 Chatbot"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Comparative Treatment Persistence in Type 2 Diabetes")
    st.subheader("Metformin vs GLP-1 RA vs SGLT-2i — Real-World Evidence Study")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Study Overview
        This real-world evidence (RWE) study characterises **comparative treatment persistence**
        across three major antidiabetic drug classes in T2DM, using Synthea-generated synthetic
        patient data structured in **OMOP CDM v5.4**.

        **Primary outcome:** Time-to-discontinuation (TTD) with a **90-day grace period** (Lim 2025).

        **15 comorbidities** are tracked as time-varying covariates (SNOMED-mapped).
        A **1:5 propensity-score matched** active-comparator new-user design is used,
        following the OHDSI LegendT2dm protocol.

        ### Study Design
        | Element | Design |
        |---------|--------|
        | Design | Active-comparator new-user cohort |
        | Matching | 1:5 PS matching (MatchIt, cobalt) |
        | Primary outcome | TTD (90-day grace, Lim 2025) |
        | Secondary outcome | Time-to-comorbidity (TTC) |
        | Statistical methods | Cox PH, KM, time-varying Cox |
        | ML | XGBoost + SHAP (1-year discontinuation) |
        | Graph | NetworkX → Neo4j Cypher |
        | Chatbot | LangChain + Groq (Llama 3.3 70B) + RAG |

        ### Pipeline Architecture
        ```
        Synthea (5,000 T2DM patients)
            ↓
        ETL → OMOP CDM DuckDB
            ↓
        3 Mutually Exclusive New-User Cohorts
            ↓
        1:5 PS Matching (R: MatchIt, cobalt)
            ├── TTD Analysis (90-day grace)
            ├── TTC Kaplan-Meier
            ├── Time-Varying Cox
            ├── TTC Cox + Forest Plot
            ├── Pearson Correlations
            ├── XGBoost + SHAP + UMAP
            └── Knowledge Graph
            ↓
        Streamlit Dashboard
        ```
        """)

    with col2:
        st.markdown("### Drug Classes")
        st.info("**Metformin** (Reference)\nFirst-line standard of care\nRxNorm: 1503297–1503301")
        st.error("**GLP-1 RA**\nSemaglutide, dulaglutide, liraglutide\nCV + weight benefit")
        st.success("**SGLT-2i**\nEmpagliflozin, dapagliflozin, canagliflozin\nCV + renal benefit")

        st.markdown("### Key Definitions")
        st.markdown("""
        - **Grace period**: 90 days (Lim 2025)
        - **Washout**: 365 days prior to index
        - **Minimum follow-up**: 90 days
        - **Cohort**: Mutually exclusive new-user
        """)

    st.divider()
    st.markdown("### References")
    refs = [
        "Iskandar IYK et al. *Br J Dermatol* 2018;178(5):1083–1094 — Drug survival methodology",
        "Marcellusi A et al. *BMJ Open* 2019;9:e024596 — T2DM persistence, Italy",
        "Lim LL et al. *Diabetologia* 2025;68(3):412–427 — 90-day grace period validation",
        "OHDSI LegendT2dm — Active-comparator new-user protocol",
        "ADA Standards of Care 2024 — *Diabetes Care* 47(Suppl 1)",
    ]
    for ref in refs:
        st.markdown(f"- {ref}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COHORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Cohort Characteristics")

    col1, col2 = st.columns(2)
    with col1:
        summary = load_csv("outputs/tables/cohort_summary.csv")
        if summary is not None:
            st.subheader("Cohort Summary")
            st.dataframe(summary.style.format(precision=2), use_container_width=True)
            for _, row in summary.iterrows():
                st.metric(
                    label=row["drug_class"].upper(),
                    value=f"n={int(row['n'])}",
                    delta=f"Median TTD: {row.get('followup_median', 'N/A'):.0f} days" if 'followup_median' in row else ""
                )
        else:
            st.info("Cohort summary not yet generated. Run `bash scripts/bootstrap.sh`.")

    with col2:
        prev = load_csv("outputs/tables/comorbidity_prevalence.csv")
        if prev is not None:
            st.subheader("Comorbidity Prevalence (%)")
            pivot = prev.pivot(index="comorbidity", columns="drug_class", values="prevalence_pct")
            st.dataframe(pivot.style.background_gradient(cmap="YlOrRd").format("{:.1f}"),
                         use_container_width=True)
        else:
            st.info("Comorbidity prevalence not yet generated.")

    st.divider()
    st.subheader("Propensity Score Matching Diagnostics")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_image("outputs/figures/ps_distribution.png", "Propensity Score Distribution")
    with col2:
        show_image("outputs/figures/love_plot_glp1_vs_met.png", "Love Plot: GLP-1 RA vs Metformin")
    with col3:
        show_image("outputs/figures/love_plot_sglt2_vs_met.png", "Love Plot: SGLT-2i vs Metformin")

    baseline = load_csv("outputs/tables/cohort_baseline.csv")
    if baseline is not None:
        st.subheader("Baseline Cohort (first 100 rows)")
        st.dataframe(baseline.head(100), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SURVIVAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Survival Analysis")

    st.subheader("Treatment Persistence — Kaplan-Meier")
    col1, col2 = st.columns(2)
    with col1:
        show_image("outputs/figures/km_ttd_by_class.png", "KM Persistence (Python/lifelines)")
    with col2:
        show_image("outputs/figures/km_persistence_survminer.png", "KM Persistence (R/survminer)")

    ttd_sum = load_csv("outputs/tables/ttd_summary.csv")
    if ttd_sum is not None:
        st.subheader("TTD Summary Statistics")
        st.dataframe(ttd_sum.style.format(precision=2), use_container_width=True)

    st.divider()
    st.subheader("Cox Proportional Hazards — Treatment Discontinuation")
    col1, col2 = st.columns([1, 1])
    with col1:
        cox = load_csv("outputs/tables/cox_ttd_results.csv")
        if cox is not None:
            st.dataframe(cox.style.format(precision=4), use_container_width=True)
    with col2:
        show_image("outputs/figures/forest_cox_ttd.png", "Forest Plot — Cox HR (TTD)")

    st.divider()
    st.subheader("Time-Varying Cox (Comorbidity 0→1 Transitions)")
    tv_cox = load_csv("outputs/tables/cox_timevarying_results.csv")
    if tv_cox is not None:
        st.dataframe(tv_cox.style.format(precision=4), use_container_width=True)
    else:
        st.info("Time-varying Cox results pending.")

    st.divider()
    st.subheader("Per-Comorbidity Stratified Kaplan-Meier (codx=0 vs codx=1)")
    show_image("outputs/figures/km_stratified_grid.png", "Stratified KM Grid (15 comorbidities)")

    km_sum = load_csv("outputs/tables/km_stratified_summary.csv")
    if km_sum is not None:
        st.dataframe(km_sum.style.format(precision=4), use_container_width=True)

    st.divider()
    st.subheader("Pearson Correlations: Comorbidity × TTD")
    col1, col2 = st.columns([1, 1])
    with col1:
        show_image("outputs/figures/correlation_heatmap.png")
    with col2:
        corr = load_csv("outputs/tables/correlations.csv")
        if corr is not None:
            st.dataframe(
                corr.sort_values("pearson_r")
                .style.format({"pearson_r": "{:.4f}", "p_value": "{:.4f}", "p_adj_bh": "{:.4f}"}),
                use_container_width=True,
            )

    st.divider()
    st.subheader("Schoenfeld Residuals (PH Assumption)")
    show_image("outputs/figures/schoenfeld_residuals.png", "Schoenfeld Residuals — Cox PH Test")

    st.divider()
    st.subheader("Time-to-Comorbidity (TTC) KM Grid")
    show_image("outputs/figures/km_ttc_grid.png", "TTC KM Grid by Drug Class")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Machine Learning — XGBoost + SHAP + UMAP")
    st.markdown("**Outcome:** 1-year treatment discontinuation (binary)  "
                "**Model:** XGBoost, 5-fold stratified CV, grid search")

    cv_res = load_csv("outputs/tables/ml_metrics.csv")
    if cv_res is not None:
        st.subheader("5-Fold Cross-Validation Results")
        st.dataframe(cv_res.style.format(precision=4), use_container_width=True)
        mean_row = cv_res[cv_res["split"] == "mean"]
        if not mean_row.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean AUROC", f"{float(mean_row['auc'].iloc[0]):.3f}")
            col2.metric("Mean F1", f"{float(mean_row['f1'].iloc[0]):.3f}")
            col3.metric("Mean Brier", f"{float(mean_row['brier'].iloc[0]):.4f}")
    else:
        st.info("ML results pending — run `bash scripts/bootstrap.sh`.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SHAP Feature Importance (Beeswarm)")
        show_image("outputs/figures/shap_beeswarm.png")
    with col2:
        st.subheader("SHAP Waterfall — Example Patients")
        show_image("outputs/figures/shap_force_examples.png")

    st.divider()
    st.subheader("UMAP of XGBoost Leaf Embeddings")
    show_image("outputs/figures/umap_phenotypes.png", "UMAP coloured by drug class")

    st.divider()
    st.subheader("Interactive Prediction")
    st.markdown("Configure a hypothetical patient to get a discontinuation probability:")
    col1, col2, col3 = st.columns(3)
    with col1:
        drug_class = st.selectbox("Drug Class", ["metformin", "glp1", "sglt2"])
        age        = st.slider("Age at Index", 18, 90, 60)
        cci        = st.slider("Charlson Comorbidity Index", 0, 10, 2)
    with col2:
        hypertension = st.checkbox("Hypertension")
        ckd          = st.checkbox("CKD")
        heart_failure = st.checkbox("Heart Failure")
        depression    = st.checkbox("Depression")
    with col3:
        obesity      = st.checkbox("Obesity")
        stroke       = st.checkbox("Stroke/TIA")
        mi           = st.checkbox("Myocardial Infarction")
        nafld        = st.checkbox("NAFLD")

    if st.button("Predict Discontinuation Risk"):
        import numpy as np, xgboost as xgb_, pandas as _pd
        model_path = Path("outputs/models/xgb_model.ubj")
        if model_path.exists():
            model = xgb_.XGBClassifier()
            model.load_model(str(model_path))
            dc_map = {"metformin": 0, "glp1": 1, "sglt2": 2}
            dc_num = dc_map[drug_class]
            comorbidity_count = sum([int(hypertension), int(obesity), int(ckd),
                                     int(heart_failure), int(depression), int(nafld),
                                     int(stroke), int(mi)])
            drug_met  = int(drug_class == "metformin")
            drug_glp1 = int(drug_class == "glp1")
            drug_sglt2 = int(drug_class == "sglt2")
            feature_vals = {
                "age_at_index": age, "age_over65": int(age >= 65),
                "sex_female": 1, "sex_male": 0,
                "drug_class_num": dc_num,
                "drug_metformin": drug_met, "drug_glp1": drug_glp1, "drug_sglt2": drug_sglt2,
                "cci": cci, "comorbidity_count": comorbidity_count,
                "days_since_t2dm_dx": 365, "followup_days": 365,
                "glp1_x_codx":  drug_glp1 * comorbidity_count,
                "sglt2_x_codx": drug_sglt2 * comorbidity_count,
                "glp1_x_cci":   drug_glp1 * cci,
                "sglt2_x_cci":  drug_sglt2 * cci,
                "hypertension": int(hypertension), "obesity": int(obesity),
                "ckd": int(ckd), "heart_failure": int(heart_failure),
                "hyperlipidemia": 0, "nash": 0, "neuropathy": 0, "retinopathy": 0,
                "depression": int(depression), "atrial_fibrillation": 0,
                "sleep_apnea": 0, "nafld": int(nafld), "pvd": 0,
                "stroke": int(stroke), "mi": int(mi),
            }
            X = _pd.DataFrame([feature_vals])
            proba = model.predict_proba(X)[0][1]
            st.metric("Estimated 1-Year Discontinuation Probability", f"{proba:.1%}")
            if proba > 0.6:
                st.warning("High discontinuation risk — consider adherence support intervention.")
            elif proba > 0.4:
                st.info("Moderate discontinuation risk.")
            else:
                st.success("Lower discontinuation risk.")
        else:
            st.error("Model not found — run `bash scripts/bootstrap.sh` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GRAPH
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Knowledge Graph: Drug — Disease — Comorbidity")
    st.markdown(
        "Directed graph encoding drug-class effects on comorbidities (TREATS), "
        "comorbidity associations with discontinuation (ASSOCIATED_WITH), "
        "and drug-class effects on TTD (DRUG_CLASS_EFFECT)."
    )

    show_image("outputs/figures/knowledge_graph.png", "T2DM Knowledge Graph (NetworkX)")

    st.divider()
    st.subheader("Cypher Export (Neo4j 5.15)")
    col1, col2 = st.columns(2)
    with col1:
        node_cypher = Path("graph/cypher_export/nodes.cypher")
        if node_cypher.exists():
            st.code(node_cypher.read_text()[:2000], language="cypher")
        else:
            st.info("Cypher not yet generated.")
    with col2:
        edge_cypher = Path("graph/cypher_export/edges.cypher")
        if edge_cypher.exists():
            st.code(edge_cypher.read_text()[:2000], language="cypher")

    st.divider()
    st.subheader("TTC Forest Plot — Drug-Class HR per Comorbidity")
    show_image("outputs/figures/forest_ttc.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Study Assistant — Llama 3.3 70B via Groq")
    st.markdown(
        "Ask questions about study results, statistical methods, or drug-class clinical profiles. "
        "The assistant has access to study outputs, the ADA 2024 guidelines, and the OMOP database."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the study..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from chatbot.chatbot import get_chatbot
                    bot = get_chatbot()
                    response = bot.get_response(prompt)
                except Exception as e:
                    response = f"Chatbot error: {e}\n\nEnsure GROQ_API_KEY is set in `.env`."
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            try:
                from chatbot.chatbot import get_chatbot
                get_chatbot().clear_history()
            except Exception:
                pass
            st.rerun()

    st.divider()
    st.markdown("**Example questions:**")
    examples = [
        "What is the median time-to-discontinuation for GLP-1 RA patients?",
        "Which comorbidities are most strongly associated with treatment discontinuation?",
        "Why is the 90-day grace period used instead of 30 days?",
        "What does the SHAP analysis show about comorbidity effects on discontinuation risk?",
        "Which drug class has the best evidence for CKD protection according to ADA 2024?",
        "How was propensity score matching performed in this study?",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")
