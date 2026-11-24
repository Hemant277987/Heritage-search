from pathlib import Path

import pandas as pd

from src.preprocess import clean_text, load_corpus
from src.search import HeritageIndex, IndexConfig

ROOT = Path(__file__).resolve().parent.parent


def _index():
    df, _ = load_corpus(ROOT / "data" / "heritage_corpus.csv")
    return HeritageIndex(IndexConfig(ngram_range=(1, 1))).fit(df)


def test_cleaning_report_matches_known_defects():
    df, rep = load_corpus(ROOT / "data" / "heritage_corpus.csv")
    assert rep["dropped_missing_text_ids"] == ["66"]
    assert rep["missing_region_filled_unknown_ids"] == ["68"]
    assert rep["missing_period_filled_unknown_ids"] == ["70"]
    assert rep["rows_with_html_markup_cleaned"] == ["69"]
    assert df["region"].notna().all() and df["period"].notna().all()
    assert "<p>" not in " ".join(df["text"])


def test_stemming_is_consistent():
    assert clean_text("dances dancing danced dance") == "danc danc danc danc"


def test_search_ranks_obvious_match_first_with_evidence():
    res = _index().search("silk saree gold thread", k=3)
    assert res[0]["category"] == "Crafts & Textiles"
    assert res[0]["evidence"], "every ranked result must carry matching terms"
    assert res[0]["score"] >= res[1]["score"]


def test_unknown_query_returns_nothing_not_noise():
    assert _index().search("xyzzy qwerty", k=5) == []


def test_category_label_is_not_a_feature():
    df, _ = load_corpus(ROOT / "data" / "heritage_corpus.csv")
    idx = HeritageIndex(IndexConfig(use_title=False)).fit(df)
    # index text must equal text_clean exactly: no title, region, period or category
    assert idx._index_text(idx.df) == idx.df["text_clean"].tolist()
