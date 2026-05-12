"""
chatbot.py — LangChain + Claude API chatbot with 3 RAG retrieval channels:
  1. SQL: DuckDB OMOP cohort queries (via SQLDatabase tool)
  2. Model loader: XGBoost predictions and SHAP explanations
  3. Document RAG: FAISS over ADA 2024 guidelines + study results

Designed as a Streamlit-embeddable chatbot; call get_response() from app.py.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import xgboost as xgb
from anthropic import Anthropic
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DB_PATH    = os.getenv("OMOP_DB_PATH",   "data/omop/omop.duckdb")
MODEL_PATH = "outputs/models/xgb_model.ubj"
ADA_PATH   = "chatbot/ada_guidelines.txt"
RESULTS_PATHS = [
    "outputs/tables/ttd_summary.csv",
    "outputs/tables/correlations.csv",
    "outputs/tables/cox_ttd_results.csv",
    "outputs/tables/cohort_summary.csv",
]

SYSTEM_PROMPT = """You are a pharmacoepidemiology research assistant for the T2DM Persistence RWE study.
You help interpret study results, explain statistical methods, and answer clinical questions about
metformin, GLP-1 receptor agonists, and SGLT-2 inhibitors in type 2 diabetes.

Study context:
- 5,000 synthetic T2DM patients (Synthea), OMOP CDM v5.4
- Primary outcome: time-to-discontinuation (90-day grace period, Lim 2025)
- 15 comorbidities tracked (SNOMED-coded)
- Methods: Cox PH, Kaplan-Meier, XGBoost + SHAP, PS matching (MatchIt)

Always cite the methodological reference when explaining a method (e.g., "per Lim 2025 for the 90-day grace period").
When you use study data, clearly indicate it comes from synthetic data only.
If asked about clinical decisions, defer to ADA 2024 guidelines and advise consultation with a clinician.
"""


class T2DMChatbot:
    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = Anthropic(api_key=api_key) if api_key else None
        self.vectorstore = self._build_vectorstore()
        self.xgb_model   = self._load_xgb_model()
        self.history: list[dict[str, str]] = []

    def _build_vectorstore(self) -> FAISS | None:
        docs = []

        # ADA guidelines
        if Path(ADA_PATH).exists():
            text = Path(ADA_PATH).read_text()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(text)
            docs.extend(chunks)

        # Study results CSVs
        for path in RESULTS_PATHS:
            if Path(path).exists():
                df = pd.read_csv(path)
                docs.append(f"Study result from {path}:\n{df.to_string(index=False)}")

        if not docs:
            log.warning("No documents for vectorstore — RAG disabled")
            return None

        # Use FakeEmbeddings (no API key required for FAISS indexing in demo mode)
        # In production, replace with langchain_anthropic or OpenAI embeddings
        embeddings = FakeEmbeddings(size=384)
        try:
            vs = FAISS.from_texts(docs, embeddings)
            log.info("Vectorstore built: %d documents", len(docs))
            return vs
        except Exception as e:
            log.warning("Vectorstore build failed: %s", e)
            return None

    def _load_xgb_model(self) -> xgb.XGBClassifier | None:
        if Path(MODEL_PATH).exists():
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            log.info("XGBoost model loaded")
            return model
        log.warning("XGBoost model not found at %s — run ml/train.py first", MODEL_PATH)
        return None

    def _query_sql(self, question: str) -> str:
        """Generate and run a DuckDB SQL query based on the question."""
        if not Path(DB_PATH).exists():
            return "OMOP database not found. Run bootstrap.sh first."
        try:
            conn = duckdb.connect(DB_PATH, read_only=True)
            # Simple heuristic routing — in production, use an LLM SQL agent
            if "cohort" in question.lower() or "how many" in question.lower():
                result = conn.execute(
                    "SELECT count(*) AS n_persons FROM person"
                ).df()
            elif "drug" in question.lower() or "exposure" in question.lower():
                result = conn.execute(
                    "SELECT drug_concept_id, count(*) AS n FROM drug_exposure GROUP BY drug_concept_id LIMIT 10"
                ).df()
            else:
                result = conn.execute(
                    "SELECT count(*) AS n_conditions FROM condition_occurrence"
                ).df()
            conn.close()
            return f"SQL result:\n{result.to_string(index=False)}"
        except Exception as e:
            return f"SQL query failed: {e}"

    def _retrieve_context(self, question: str) -> str:
        parts = []

        # RAG retrieval
        if self.vectorstore:
            try:
                docs = self.vectorstore.similarity_search(question, k=3)
                parts.append("Relevant context:\n" + "\n---\n".join(d.page_content for d in docs))
            except Exception as e:
                log.debug("RAG retrieval failed: %s", e)

        # SQL channel (triggered by data questions)
        keywords = ["how many", "count", "number of", "patients", "cohort size", "database"]
        if any(k in question.lower() for k in keywords):
            parts.append(self._query_sql(question))

        # XGBoost channel (triggered by prediction questions)
        if self.xgb_model and any(k in question.lower() for k in ["predict", "risk", "probability", "shap"]):
            parts.append(
                "XGBoost model is loaded. 1-year discontinuation prediction is available. "
                "SHAP feature importances are in outputs/figures/shap_beeswarm.png."
            )

        return "\n\n".join(parts) if parts else ""

    def get_response(self, user_message: str) -> str:
        if self.client is None:
            return (
                "ANTHROPIC_API_KEY not set. Add it to your .env file to enable the chatbot. "
                "Study results are available in the Survival, ML, and Graph tabs."
            )

        context = self._retrieve_context(user_message)

        system = SYSTEM_PROMPT
        if context:
            system += f"\n\nRelevant context for this query:\n{context}"

        self.history.append({"role": "user", "content": user_message})

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system,
                messages=self.history,
            )
            assistant_message = response.content[0].text
            self.history.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            log.error("Claude API error: %s", e)
            return f"API error: {e}"

    def clear_history(self) -> None:
        self.history = []


# Module-level singleton for Streamlit (cached across reruns)
_chatbot_instance: T2DMChatbot | None = None


def get_chatbot() -> T2DMChatbot:
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = T2DMChatbot()
    return _chatbot_instance
