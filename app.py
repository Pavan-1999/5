import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
from rank_bm25 import BM25Okapi

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Capstone Similarity Checker",
    page_icon="🔎",
    layout="wide",
)
st.title("🔎 Capstone Similarity Checker")
st.caption(
    "Live Google Sheet → BM25 lexical gate + E5 semantic embeddings → "
    "Top matches with interpretable similarity scores."
)

# ===================== DEFAULTS =====================
DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQOt3ScW1TkCpKVCP2vNMbNSahbMkaFZBARjoTRe267tQdX_E_hC8o3bXTjwkhPxdXKtKfq1_dWLZMU/pub?gid=1929751519&single=true&output=csv"
)

PREFERRED_TITLE_NAMES = [
    "Project Title",
    "Title",
    "Capstone Title",
    "title",
    "project_title",
    "Project",
]


# ===================== TEXT HELPERS =====================
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenize(text: str):
    # very simple tokenization: lowercase, remove non-letters, split on spaces
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def expand_query(q: str) -> str:
    """Small typo fixes + expansions for common abbreviations."""
    q2 = normalize_text(q)
    fixes = {
        "artifical": "artificial",
        "intellegence": "intelligence",
        "machne": "machine",
        "lern": "learning",
    }
    for wrong, right in fixes.items():
        q2 = q2.replace(wrong, right)

    # simple domain-related expansions
    if q2 == "ai" or " ai " in f" {q2} ":
        q2 += " artificial intelligence"
    if q2 == "ml" or " ml " in f" {q2} ":
        q2 += " machine learning"
    if "nlp" in q2:
        q2 += " natural language processing"

    return q2


def strength_label(score: float) -> str:
    """Convert hybrid score in [0,1] to a verbal band."""
    if score < 0.30:
        return "Weak"
    elif score < 0.60:
        return "Moderate"
    elif score < 0.80:
        return "Strong"
    else:
        return "Very strong"


# ===================== MODEL LOADERS =====================
@st.cache_resource(show_spinner=False)
def load_e5_model():
    """
    Load the E5 embedding model (better for short text than vanilla SBERT).
    """
    from sentence_transformers import SentenceTransformer

    # E5-small is good balance of quality + speed
    return SentenceTransformer("intfloat/e5-small-v2")


@st.cache_data(show_spinner=False)
def build_bm25_index(titles: pd.Series):
    """
    Build BM25 index over tokenized titles.
    """
    tokenized_corpus = [tokenize(t) for t in titles.astype(str).tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


@st.cache_data(show_spinner=False)
def embed_titles_e5(titles: pd.Series):
    """
    Encode all existing titles once with E5.
    For best performance, E5 expects 'passage: ' prefix for documents.
    """
    model = load_e5_model()
    docs = ["passage: " + normalize_text(t) for t in titles.astype(str).tolist()]
    embs = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs)


# ===================== DATA HELPERS =====================
@st.cache_data(show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


@st.cache_data(show_spinner=False)
def detect_title_column(df: pd.DataFrame) -> str:
    for c in PREFERRED_TITLE_NAMES:
        if c in df.columns:
            return c
    # fallback: first text-like column
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            return c
    raise ValueError("No suitable text column found for titles.")


@st.cache_data(show_spinner=False)
def clean_titles(series: pd.Series) -> pd.Series:
    s = series.astype(str).fillna("").str.strip()
    s = s[s.str.len() > 0].reset_index(drop=True)
    return s


# ===================== CORE SIMILARITY ENGINE =====================
def find_similar_titles_e5_bm25(
    query: str,
    titles: pd.Series,
    bm25: BM25Okapi,
    tokenized_corpus,
    title_embs: np.ndarray,
    top_k: int = 5,
    bm25_candidate_frac: float = 0.30,  # relative gate: 30% of max BM25
    w_semantic: float = 0.70,
    w_lexical: float = 0.30,
):
    """
    Lexical gate (BM25) + E5 semantic embeddings.
    - We first score all titles lexically using BM25.
    - We keep only candidates whose BM25 score is at least
      bm25_candidate_frac * max_bm25.
    - On these candidates, we compute E5 semantic similarity and a hybrid score.

    Returns:
      results_df: DataFrame with matches and scores
      info: dict with summary flags / messages
    """
    q = expand_query(query)
    if not q:
        return None, {"error": "Please enter a non-empty title."}

    q_tokens = tokenize(q)
    if not q_tokens:
        return None, {"error": "Query has no usable words after cleaning."}

    # ---------- LEXICAL SIMILARITY (BM25) ----------
    bm25_scores = np.array(bm25.get_scores(q_tokens))  # higher = more similar
    max_bm25 = float(bm25_scores.max())

    info = {
        "max_bm25_raw": max_bm25,
    }

    if max_bm25 <= 0:
        info["message"] = (
            "No titles share enough wording with the query. "
            "We consider this title lexically novel in this dataset."
        )
        empty = pd.DataFrame(
            columns=[
                "Existing Title",
                "BM25 Raw",
                "Lexical (0–1)",
                "Semantic (0–1)",
                "Hybrid (0–1)",
                "Hybrid %",
                "Strength",
            ]
        )
        return empty, info

    # Relative lexical gate: candidate if BM25 >= fraction of max
    bm25_gate = bm25_candidate_frac * max_bm25
    cand_mask = bm25_scores >= bm25_gate
    cand_idx = np.where(cand_mask)[0]

    info["bm25_gate"] = bm25_gate
    info["num_candidates"] = int(len(cand_idx))

    if len(cand_idx) == 0:
        info["message"] = (
            "BM25 scores are very low for all titles. "
            "We consider this title lexically novel in this dataset."
        )
        empty = pd.DataFrame(
            columns=[
                "Existing Title",
                "BM25 Raw",
                "Lexical (0–1)",
                "Semantic (0–1)",
                "Hybrid (0–1)",
                "Hybrid %",
                "Strength",
            ]
        )
        return empty, info

    # Normalize BM25 to [0,1] for interpretability: score / max_score
    lex_norm = np.clip(bm25_scores[cand_idx] / max_bm25, 0.0, 1.0)

    # ---------- SEMANTIC SIMILARITY (E5) ----------
    model = load_e5_model()
    q_emb = model.encode(
        ["query: " + normalize_text(q)],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    cand_embs = title_embs[cand_idx]
    # dot product of normalized vectors = cosine similarity in [-1,1]
    sem_raw = np.dot(cand_embs, q_emb)
    sem_norm = np.clip((sem_raw + 1.0) / 2.0, 0.0, 1.0)  # -> [0,1]

    # ---------- HYBRID SCORE ----------
    hybrid = w_semantic * sem_norm + w_lexical * lex_norm

    # ---------- RANK & FORMAT RESULTS ----------
    order = np.argsort(-hybrid)
    k = min(int(top_k), len(order))
    sel = order[:k]
    idx_sel = cand_idx[sel]

    results = pd.DataFrame(
        {
            "Existing Title": titles.iloc[idx_sel].values,
            "BM25 Raw": np.round(bm25_scores[idx_sel], 4),
            "Lexical (0–1)": np.round(lex_norm[sel], 4),
            "Semantic (0–1)": np.round(sem_norm[sel], 4),
            "Hybrid (0–1)": np.round(hybrid[sel], 4),
        }
    )
    results["Hybrid %"] = (results["Hybrid (0–1)"] * 100).round(2)
    results["Strength"] = results["Hybrid (0–1)"].apply(strength_label)

    if max_bm25 < 1.0:
        info["message"] = (
            "Lexical overlap is modest. These are the closest titles, "
            "but the new title may still represent a relatively novel topic."
        )
    else:
        info["message"] = (
            "Some titles share meaningful wording with the query. "
            "Review matches labelled 'Strong' or 'Very strong' for potential overlap."
        )

    return results, info


# ===================== UI: DATA SOURCE =====================
with st.expander("📄 Data Source (Google Sheet CSV)"):
    sheet_url = st.text_input(
        "Paste your **published-to-web CSV** link (must end with `output=csv`):",
        value=DEFAULT_SHEET_URL,
    )
    st.caption("Google Sheets → File → Share → Publish to web → CSV → copy the link.")

df = None
err_box = st.empty()
try:
    df = fetch_csv(sheet_url)
except Exception as e:
    err_box.error(
        "Could not load CSV. Ensure the link is public & ends with `output=csv`.\n\n"
        f"Error: {e}"
    )

if df is not None:
    try:
        title_col = detect_title_column(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    titles = clean_titles(df[title_col])
    if titles.empty:
        st.error("No non-empty titles found in the detected column.")
        st.stop()

    st.markdown("### Dataset overview")
    col_a, col_b = st.columns(2)
    col_a.metric("Total projects loaded", len(titles))

    year_col = None
    for cand in ["Year", "year", "Academic Year", "Year_of_Completion"]:
        if cand in df.columns:
            year_col = cand
            break

    if year_col:
        year_counts = df[year_col].value_counts().sort_index()
        col_b.bar_chart(year_counts, use_container_width=True)
        col_b.caption(f"Projects per {year_col}")
    else:
        col_b.write("Add a 'Year' column in your sheet to see projects per year.")

    with st.expander("Preview (first 5 titles)"):
        st.write(pd.DataFrame(titles.head(5), columns=["Title"]))

    # Build indices once (cached)
    bm25, tokenized_corpus = build_bm25_index(titles)
    title_embs = embed_titles_e5(titles)

    # ===================== UI: QUERY & RESULTS =====================
    st.markdown("### Check a new capstone title")
    qcol1, qcol2 = st.columns([2, 1])
    with qcol1:
        query_title = st.text_input(
            "Enter a new capstone title to check:",
            placeholder="e.g., AI-based demand forecasting for retail supply chains",
        )
    with qcol2:
        top_k = st.number_input(
            "Top matches", min_value=1, max_value=50, value=5, step=1
        )

    if st.button("Check Similarity", type="primary", use_container_width=True):
        results_df, info = find_similar_titles_e5_bm25(
            query_title,
            titles,
            bm25,
            tokenized_corpus,
            title_embs,
            top_k=int(top_k),
        )

        if "error" in info:
            st.warning(info["error"])
        else:
            st.markdown(f"**Summary:** {info.get('message', '')}")
            st.caption(
                f"Max BM25 score in dataset for this query: {info.get('max_bm25_raw', 0):.3f} "
                f"(candidates after gate: {info.get('num_candidates', 0)})"
            )

            if results_df.empty:
                st.info("No candidate titles to display.")
            else:
                st.subheader("Top candidate matches")
                st.dataframe(results_df, use_container_width=True)

                st.download_button(
                    "⬇️ Download results (CSV)",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="similarity_results.csv",
                    mime="text/csv",
                )

                with st.expander("How to interpret the scores"):
                    st.markdown(
                        """
                        - **BM25 Raw**: lexical score from the BM25 algorithm. Higher = more word overlap.
                        - **Lexical (0–1)**: BM25 score normalized by the best score for this query.
                        - **Semantic (0–1)**: E5 embedding cosine similarity (mapped from [-1,1] to [0,1]).
                        - **Hybrid (0–1)**: 0.7 × Semantic + 0.3 × Lexical. Used for the **Hybrid %** and **Strength** label.  

                        Rough guideline for **Hybrid** / **Hybrid %**:

                        - < 0.30 (~0–30%) → **Weak** similarity  
                        - 0.30–0.60 (~30–60%) → **Moderate** topic overlap  
                        - 0.60–0.80 (~60–80%) → **Strong** overlap (investigate)  
                        - > 0.80 (>80%) → **Very strong**; likely duplicate or very close topic  

                        The tool is designed to **assist faculty** in spotting potential overlaps.  
                        Final decisions should always be made by supervisors using academic judgement.
                        """
                    )

st.markdown("---")
st.caption(
    "Tip: Good titles specify domain + method + context "
    "(e.g., 'AI-based demand forecasting for Canadian grocery retail')."
)
