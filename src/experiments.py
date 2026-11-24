"""Repeatable experiments for objectives 2 and 3. Run: python run_experiments.py"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix,
                             f1_score, normalized_mutual_info_score, silhouette_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocess import leakage_audit, load_corpus, save_json
from .search import HeritageIndex, IndexConfig

SEED = 42


# ----------------------------------------------------------------------------
# Objective 2: classical model comparison (target = curated heritage category)
# ----------------------------------------------------------------------------
def make_models():
    return {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Multinomial NB": MultinomialNB(alpha=0.1),
        "Complement NB": ComplementNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=10, max_iter=2000),
        "Linear SVM": LinearSVC(C=1.0),
        "kNN (cosine, k=5)": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
    }


def classification_experiment(df: pd.DataFrame) -> dict:
    X, y = df["text_clean"], df["category"]
    # Hold-out split is made BEFORE any fitting; the vectoriser lives inside the
    # Pipeline so TF-IDF statistics are learned from training folds only.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=5, random_state=SEED)

    rows = []
    fold_scores = {}
    for ngram in [(1, 1), (1, 2)]:
        for name, model in make_models().items():
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=ngram, sublinear_tf=True,
                                          token_pattern=r"\S+", lowercase=False)),
                ("clf", model),
            ])
            res = cross_validate(pipe, X_tr, y_tr, cv=cv,
                                 scoring={"acc": "accuracy", "f1": "f1_macro"})
            fold_scores[(name, ngram)] = res["test_f1"]
            rows.append({
                "model": name, "ngram": f"{ngram[0]}-{ngram[1]}",
                "cv_accuracy": res["test_acc"].mean(), "cv_accuracy_sd": res["test_acc"].std(),
                "cv_macro_f1": res["test_f1"].mean(), "cv_macro_f1_sd": res["test_f1"].std(),
            })
    table = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)

    # Paired effect of the one feature variation (unigram vs uni+bigram), same folds.
    variation = []
    for name in make_models():
        d = fold_scores[(name, (1, 2))] - fold_scores[(name, (1, 1))]
        variation.append({"model": name,
                          "mean_macro_f1_gain_of_bigrams": float(d.mean()),
                          "sd": float(d.std()),
                          "folds_bigram_better": int((d > 0).sum()),
                          "folds_unigram_better": int((d < 0).sum()),
                          "n_folds": int(len(d))})

    # Preferred model = best CV macro-F1; touched on the hold-out set exactly once.
    best = table.iloc[0]
    ngram = tuple(int(v) for v in best["ngram"].split("-"))
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram, sublinear_tf=True,
                                  token_pattern=r"\S+", lowercase=False)),
        ("clf", make_models()[best["model"]]),
    ]).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    labels = sorted(y.unique())
    holdout = {
        "model": best["model"], "ngram": best["ngram"],
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_te, pred, labels=labels).tolist(),
        "errors": [
            {"title": df.loc[i, "title"], "true": t, "predicted": p}
            for i, t, p in zip(X_te.index, y_te, pred) if t != p
        ],
    }
    return {"cv_table": table.round(4).to_dict("records"),
            "variation_ngram": variation, "holdout": holdout}


# ----------------------------------------------------------------------------
# K-Means topic discovery (unsupervised; labels used only for external scoring)
# ----------------------------------------------------------------------------
def clustering_experiment(df: pd.DataFrame) -> dict:
    vec = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True,
                          token_pattern=r"\S+", lowercase=False)
    X = vec.fit_transform(df["text_clean"])
    terms = np.array(vec.get_feature_names_out())
    sweep = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        sweep.append({"k": k, "silhouette": float(silhouette_score(X, km.labels_)),
                      "inertia": float(km.inertia_)})
    k = df["category"].nunique()
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    order = km.cluster_centers_.argsort(axis=1)[:, ::-1]
    clusters = []
    for c in range(k):
        members = df[km.labels_ == c]
        top = Counter(members["category"]).most_common(1)[0]
        clusters.append({"cluster": c, "size": int(len(members)),
                         "top_terms": terms[order[c, :8]].tolist(),
                         "majority_category": top[0],
                         "purity": top[1] / len(members),
                         "titles": members["title"].tolist()})
    purity = sum(c["purity"] * c["size"] for c in clusters) / len(df)
    return {"k_sweep": sweep, "k_used": k,
            "ARI": float(adjusted_rand_score(df["category"], km.labels_)),
            "NMI": float(normalized_mutual_info_score(df["category"], km.labels_)),
            "purity": float(purity), "clusters": clusters}


# ----------------------------------------------------------------------------
# Objective 3: top-k relevance / ranking quality
# ----------------------------------------------------------------------------
def _relevance(df: pd.DataFrame, cats: set[str]) -> np.ndarray:
    return df["category"].isin(cats).to_numpy().astype(int)


def ranking_metrics(ranked_rel: np.ndarray, n_rel: int) -> dict:
    p5 = ranked_rel[:5].mean()
    hits = np.where(ranked_rel == 1)[0]
    mrr = 1.0 / (hits[0] + 1) if len(hits) else 0.0
    gains = ranked_rel[:10] / np.log2(np.arange(2, 12)[: len(ranked_rel[:10])])
    ideal = np.ones(min(n_rel, 10)) / np.log2(np.arange(2, 2 + min(n_rel, 10)))
    ndcg = gains.sum() / ideal.sum() if ideal.sum() else 0.0
    return {"P@5": float(p5), "MRR": float(mrr), "nDCG@10": float(ndcg)}


def keyword_overlap_scores(index: HeritageIndex, query: str) -> np.ndarray:
    """Naive baseline: number of distinct query terms present in each document."""
    q = set(index.query_vector(query).nonzero()[1])
    binary = (index.X > 0).astype(int)
    return np.asarray(binary[:, list(q)].sum(axis=1)).ravel() if q else np.zeros(index.X.shape[0])


def evaluate_ranker(index: HeritageIndex, queries: pd.DataFrame, scorer=None) -> pd.DataFrame:
    rows = []
    for q in queries.itertuples():
        cats = set(q.relevant_categories.split("|"))
        rel = _relevance(index.df, cats)
        s = index.scores(q.query) if scorer is None else scorer(index, q.query)
        # stable, deterministic tie-break by document order
        order = np.argsort(-s, kind="stable")
        ranked = rel[order]
        # zero-score documents carry no evidence; treat them as non-retrieved
        ranked = np.where(s[order] > 0, ranked, 0)
        m = ranking_metrics(ranked, int(rel.sum()))
        rows.append({"qid": q.qid, "split": q.split, "query": q.query, **m})
    return pd.DataFrame(rows)


def retrieval_experiment(df: pd.DataFrame, queries: pd.DataFrame) -> dict:
    grid = list(itertools.product([(1, 1), (1, 2)], [False, True], [False, True]))
    rows, per_query = [], {}
    for ngram, stem, use_title in grid:
        # stemming changes text_clean, so rebuild it for each config
        from .preprocess import clean_text
        d = df.copy()
        d["text_clean"] = d["text"].map(lambda t: clean_text(t, stem=stem))
        idx = HeritageIndex(IndexConfig(ngram_range=ngram, stem=stem,
                                        use_title=use_title)).fit(d)
        res = evaluate_ranker(idx, queries)
        name = f"ngram={ngram[0]}-{ngram[1]}, stem={stem}, title={use_title}"
        per_query[name] = res
        for split in ["dev", "test"]:
            r = res[res["split"] == split]
            rows.append({"config": name, "split": split, **r[["P@5", "MRR", "nDCG@10"]].mean().to_dict()})
    table = pd.DataFrame(rows)

    # Baseline with the same index as the preferred config (chosen below on DEV only).
    dev = table[table["split"] == "dev"].sort_values(["nDCG@10", "P@5"], ascending=False)
    best_name = dev.iloc[0]["config"]

    # Rebuild best config for baseline + failure inspection.
    parts = dict(p.split("=") for p in best_name.replace(" ", "").split(","))
    ngram = tuple(int(v) for v in parts["ngram"].split("-"))
    stem, use_title = parts["stem"] == "True", parts["title"] == "True"
    from .preprocess import clean_text
    d = df.copy()
    d["text_clean"] = d["text"].map(lambda t: clean_text(t, stem=stem))
    best_idx = HeritageIndex(IndexConfig(ngram_range=ngram, stem=stem, use_title=use_title)).fit(d)
    base = evaluate_ranker(best_idx, queries, scorer=keyword_overlap_scores)
    baseline_rows = [{"split": s, **base[base["split"] == s][["P@5", "MRR", "nDCG@10"]].mean().to_dict()}
                     for s in ["dev", "test"]]

    failures = inspect_failures(best_idx, queries, per_query[best_name])
    return {"table": table.round(4).to_dict("records"), "selected_on_dev": best_name,
            "keyword_overlap_baseline": baseline_rows,
            "per_query": per_query[best_name].round(4).to_dict("records"),
            "failures": failures, "_index": best_idx}


def inspect_failures(index: HeritageIndex, queries: pd.DataFrame, per_query: pd.DataFrame,
                     threshold: float = 0.8) -> list[dict]:
    """For every query with P@5 below threshold list wrong hits and explain why."""
    cat_of_term = defaultdict(Counter)
    for text, cat in zip(index._index_text(index.df), index.df["category"]):
        for t in set(text.split()):
            cat_of_term[t][cat] += 1
    out = []
    for q in queries.itertuples():
        pq = per_query[per_query["qid"] == q.qid].iloc[0]
        if pq["P@5"] >= threshold:
            continue
        cats = set(q.relevant_categories.split("|"))
        top = index.search(q.query, k=5)
        wrong = []
        for r in top:
            if r["category"] in cats:
                continue
            ev = r["evidence"][0]["term"] if r["evidence"] else None
            wrong.append({
                "title": r["title"], "category": r["category"], "score": round(r["score"], 3),
                "matched_terms": [e["term"] for e in r["evidence"]],
                "top_term_category_spread": dict(cat_of_term[ev]) if ev else {},
            })
        out.append({"qid": int(q.qid), "query": q.query, "relevant": sorted(cats),
                    "P@5": pq["P@5"], "MRR": pq["MRR"],
                    "oov_terms": index.oov_terms(q.query),
                    "n_results_with_any_evidence": len(top),
                    "top5": [{"title": r["title"], "category": r["category"],
                              "score": round(r["score"], 3)} for r in top],
                    "wrong_hits": wrong})
    return out


# ----------------------------------------------------------------------------
def run_all(root: Path) -> dict:
    df, report = load_corpus(root / "data" / "heritage_corpus.csv")
    queries = pd.read_csv(root / "data" / "queries.csv")
    out = {"cleaning": report, "leakage_audit": leakage_audit(df)}
    out["classification"] = classification_experiment(df)
    out["clustering"] = clustering_experiment(df)
    retr = retrieval_experiment(df, queries)
    retr.pop("_index")
    out["retrieval"] = retr
    out["n_queries"] = {"dev": int((queries.split == "dev").sum()),
                        "test": int((queries.split == "test").sum())}
    save_json(out, root / "outputs" / "results.json")
    return out
