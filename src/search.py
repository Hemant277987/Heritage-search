"""TF-IDF + cosine-similarity search with matching-term evidence, plus K-Means topics."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import clean_text, light_stem

_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class IndexConfig:
    ngram_range: tuple = (1, 2)
    sublinear_tf: bool = True
    stem: bool = True
    use_title: bool = True   # index title + text (search only; never used as a model feature)
    min_df: int = 1
    n_clusters: int = 6
    random_state: int = 42


@dataclass
class HeritageIndex:
    config: IndexConfig = field(default_factory=IndexConfig)

    def fit(self, df: pd.DataFrame) -> "HeritageIndex":
        cfg = self.config
        self.df = df.reset_index(drop=True)
        docs = self._index_text(self.df)
        self.vec = TfidfVectorizer(
            ngram_range=cfg.ngram_range,
            sublinear_tf=cfg.sublinear_tf,
            min_df=cfg.min_df,
            lowercase=False,
            token_pattern=r"\S+",
        )
        self.X = self.vec.fit_transform(docs)
        self.terms = np.array(self.vec.get_feature_names_out())
        self.vocab = self.vec.vocabulary_

        # Unsupervised topic discovery (no labels used).
        self.km = KMeans(n_clusters=cfg.n_clusters, n_init=20, random_state=cfg.random_state)
        self.df["cluster"] = self.km.fit_predict(self.X)
        self.cluster_terms = self._cluster_top_terms(8)
        return self

    # ---- helpers -----------------------------------------------------------
    def _index_text(self, df: pd.DataFrame) -> list[str]:
        if self.config.use_title:
            titles = df["title"].map(lambda t: clean_text(t, stem=self.config.stem))
            return (titles + " " + df["text_clean"]).tolist()
        return df["text_clean"].tolist()

    def _cluster_top_terms(self, n: int) -> dict[int, list[str]]:
        order = self.km.cluster_centers_.argsort(axis=1)[:, ::-1]
        return {int(c): self.terms[order[c, :n]].tolist() for c in range(order.shape[0])}

    def query_vector(self, query: str):
        return self.vec.transform([clean_text(query, stem=self.config.stem)])

    def scores(self, query: str) -> np.ndarray:
        q = self.query_vector(query)
        # rows are L2-normalised, so the dot product is the cosine similarity
        return (self.X @ q.T).toarray().ravel()

    # ---- main API ----------------------------------------------------------
    def search(self, query: str, k: int = 5, category: str | None = None,
               min_score: float = 0.0) -> list[dict]:
        q = self.query_vector(query)
        sims = (self.X @ q.T).toarray().ravel()
        order = np.argsort(-sims, kind="stable")
        out = []
        for i in order:
            if sims[i] <= min_score:
                break  # zero overlap means no lexical evidence; do not pad with noise
            row = self.df.iloc[i]
            if category and row["category"] != category:
                continue
            out.append({
                "rank": len(out) + 1,
                "doc_id": row["doc_id"],
                "title": row["title"],
                "category": row["category"],
                "region": row["region"],
                "period": row["period"],
                "score": float(sims[i]),
                "cluster": int(row["cluster"]),
                "cluster_terms": self.cluster_terms[int(row["cluster"])][:5],
                "evidence": self.evidence(q, i),
                "snippet": self.snippet(row["text"], query),
                "text": row["text"],
            })
            if len(out) >= k:
                break
        return out

    def evidence(self, q, doc_idx: int, n: int = 5) -> list[dict]:
        """Terms that contributed most to the cosine score (q_tfidf * d_tfidf)."""
        contrib = q.multiply(self.X[doc_idx]).tocoo()
        pairs = sorted(zip(contrib.col, contrib.data), key=lambda p: -p[1])[:n]
        total = float(contrib.data.sum()) or 1.0
        return [{"term": self.terms[c], "share": float(v / total)} for c, v in pairs]

    @staticmethod
    def snippet(text: str, query: str) -> str:
        """Best-matching sentence (extractive, used as the default summary)."""
        qtok = {light_stem(t) for t in clean_text(query, stem=False).split()}
        sents = [s for s in _SENT.split(text) if s]
        if not sents:
            return text
        def score(s):
            toks = {light_stem(t) for t in clean_text(s, stem=False).split()}
            return len(qtok & toks)
        return max(sents, key=score)

    def oov_terms(self, query: str) -> list[str]:
        """Query terms (after cleaning) that never occur in the corpus vocabulary."""
        return [t for t in clean_text(query, stem=self.config.stem).split()
                if t not in self.vocab]
