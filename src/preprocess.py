"""Cleaning, missing-value treatment and feature construction.

Everything here is deterministic and documented in the returned report so the
same steps can be repeated for any new heritage corpus.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)
_TAG = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACE = re.compile(r"\s+")

REQUIRED_COLUMNS = ["doc_id", "title", "region", "period", "category", "text"]
MIN_TOKENS = 8  # shorter than this is not a usable document


def light_stem(token: str) -> str:
    """Very small suffix stripper (no external data needed).

    It only has to be *consistent* between documents and queries so that
    "dance", "dances", "dancing" and "danced" share one feature.
    """
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 6 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 5 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 5 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def clean_text(text: str, stem: bool = True) -> str:
    """HTML/entity removal, lowercase, letters only, stop words, optional stem."""
    text = html.unescape(str(text))
    text = _TAG.sub(" ", text)
    text = text.replace("\xa0", " ").lower()
    text = _NON_ALPHA.sub(" ", text)
    tokens = [t for t in _SPACE.split(text) if len(t) > 2 and t not in STOP_WORDS]
    if stem:
        tokens = [light_stem(t) for t in tokens]
    return " ".join(tokens)


def clean_dataframe(raw: pd.DataFrame, stem: bool = True):
    """Return (clean_df, report). Each rule is applied in a fixed, logged order."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"corpus is missing columns: {missing_cols}")

    report: dict = {"rows_raw": int(len(raw))}
    df = raw.copy()

    # 1. Normalise raw text (HTML tags, entities, whitespace) but keep casing/punctuation
    #    so the display text stays readable.
    def _display(t):
        if pd.isna(t):
            return ""
        t = _TAG.sub(" ", html.unescape(str(t))).replace("\xa0", " ")
        return _SPACE.sub(" ", t).strip()

    had_markup = df["text"].fillna("").astype(str).str.contains(r"<[^>]+>|&\w+;", regex=True)
    report["rows_with_html_markup_cleaned"] = df.loc[had_markup, "doc_id"].tolist()
    df["text"] = df["text"].map(_display)

    # 2. Missing text cannot be imputed (it *is* the data) -> drop and log ids.
    empty = df["text"].str.len() == 0
    report["dropped_missing_text_ids"] = df.loc[empty, "doc_id"].tolist()
    df = df.loc[~empty]

    # 3. Missing categorical metadata -> explicit "Unknown" level (kept, not dropped).
    for col in ["region", "period"]:
        miss = df[col].isna() | (df[col].astype(str).str.strip() == "")
        report[f"missing_{col}_filled_unknown_ids"] = df.loc[miss, "doc_id"].tolist()
        df.loc[miss, col] = "Unknown"

    # 4. The target must never be imputed.
    bad_target = df["category"].isna()
    report["dropped_missing_target_ids"] = df.loc[bad_target, "doc_id"].tolist()
    df = df.loc[~bad_target]

    # 5. Duplicates: same title + same text, or same text after normalisation.
    before = len(df)
    df = df.drop_duplicates(subset=["title", "text"], keep="first")
    df = df.drop_duplicates(subset=["text"], keep="first")
    report["dropped_duplicate_rows"] = int(before - len(df))

    # 6. Feature construction.
    df["text_clean"] = df["text"].map(lambda t: clean_text(t, stem=stem))
    df["n_tokens"] = df["text_clean"].str.split().str.len()
    short = df["n_tokens"] < MIN_TOKENS
    report["dropped_too_short_ids"] = df.loc[short, "doc_id"].tolist()
    df = df.loc[~short].reset_index(drop=True)

    report["rows_clean"] = int(len(df))
    report["class_counts"] = df["category"].value_counts().to_dict()
    report["tokens_per_doc"] = {
        "min": int(df["n_tokens"].min()),
        "median": float(df["n_tokens"].median()),
        "max": int(df["n_tokens"].max()),
    }
    return df, report


def leakage_audit(df: pd.DataFrame) -> dict:
    """Check whether label words appear verbatim in the model-input text.

    The model input is text_clean only. title, region, period, doc_id and the
    category itself are never features. This audit reports any label-word
    overlap so it is documented rather than assumed absent.
    """
    label_words = {
        "Monuments & Architecture": {"monument", "architectur"},
        "Festivals & Rituals": {"festival", "ritual"},
        "Crafts & Textiles": {"craft", "textil"},
        "Music & Dance": {"music", "danc"},
        "Cuisine & Foodways": {"cuisin", "foodway"},
        "Natural & Sacred Landscapes": {"natural", "landscap"},
    }
    hits = {}
    for cat, words in label_words.items():
        sub = df[df["category"] == cat]
        n = sum(any(w in set(t.split()) for w in words) for t in sub["text_clean"])
        hits[cat] = {"docs_containing_label_word": int(n), "docs": int(len(sub))}
    return {
        "features_used": ["text_clean"],
        "excluded_from_features": ["doc_id", "title", "region", "period", "category"],
        "label_word_overlap": hits,
    }


def load_corpus(path: str | Path, stem: bool = True):
    raw = pd.read_csv(path, dtype={"doc_id": str})
    return clean_dataframe(raw, stem=stem)


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
