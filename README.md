# Digital Heritage Search and Knowledge Discovery

Ranked search over heritage texts using NLP preprocessing, TF-IDF, cosine similarity and K-Means, with an optional LLM summary. Every result shows its similarity score, the exact query terms that produced it, and a discovered topic group.

## Run it

```bash
pip install -r requirements.txt
python run_experiments.py     # reproduces every number below -> outputs/results.json
python -m pytest -q tests     # 5 tests
python app.py                 # http://127.0.0.1:5000
```

Optional LLM summaries: `export ANTHROPIC_API_KEY=...` (model via `HERITAGE_LLM_MODEL`, default `claude-haiku-5-5`). Without a key the "Summary" checkbox shows an extractive summary (original sentences from the top results). The search never depends on the network.

## Project map

| Path | Purpose |
|---|---|
| `data/heritage_corpus.csv` | 70 rows: `doc_id,title,region,period,category,text` |
| `data/queries.csv` | 30 search queries with relevant categories, split 18 dev / 12 test |
| `src/preprocess.py` | cleaning, missing-value rules, feature construction, leakage audit |
| `src/search.py` | TF-IDF index, cosine ranking, evidence terms, K-Means topics |
| `src/experiments.py` | model comparison, clustering, ranking evaluation, failure inspection |
| `src/llm.py` | optional Anthropic API summary with extractive fallback |
| `app.py`, `templates/` | web interface (`/`, `/evaluation`, `/api/search`, `/health`) |

## Objective 1: data preparation

Cleaning rules run in a fixed order and are logged in `outputs/results.json` (`cleaning`):

- HTML tags and entities removed (1 row), whitespace normalised.
- Missing **text**: row dropped, because text cannot be imputed (1 row, id 66).
- Missing **region / period**: filled with an explicit `Unknown` level (ids 68, 70). These are display metadata only.
- Missing **target** is never imputed (none found). Exact duplicates are dropped (none after the empty row).
- Text features: lowercase, letters only, stop-word removal, light suffix stemmer; `n_tokens` constructed. 69 clean documents.

**Target:** curated `category` (6 classes), a supervised label used only to compare classifiers and to define search relevance.

**Leakage controls:** the model input is `text_clean` only; `title`, `region`, `period`, `doc_id` and `category` are excluded. The hold-out split is made before any fitting, and TF-IDF sits inside a scikit-learn `Pipeline`, so it is fitted on training folds only. Retrieval configuration is chosen on the 18 dev queries and reported on the 12 test queries. Seeds are fixed (`SEED=42`).

## Objective 2: model comparison and the feature variation

Repeated stratified CV (4 folds x 5 repeats) on the training split, macro-F1:

| Model (unigrams) | Macro-F1 |
|---|---|
| Linear SVM | 0.896 ± 0.086 |
| Complement NB | 0.895 ± 0.083 |
| Multinomial NB | 0.888 ± 0.083 |
| Logistic Regression, kNN, baseline | see `outputs/results.json` |

**Variation tested:** unigrams vs unigrams+bigrams, paired on identical folds. Bigrams did not help: the mean macro-F1 change was negative for the SVM (-0.007), Multinomial NB (-0.018), Complement NB (-0.017) and Logistic Regression (-0.025), and roughly neutral for kNN (+0.006). With ~20 words per document, bigrams mostly add sparse, rarely repeated features.

**Preferred approach:** Linear SVM on unigram TF-IDF. Caveat: the top three models are within one standard deviation of each other, so this is a preference by a narrow margin, not a proven win. Hold-out test (14 documents, used once): accuracy 0.857, macro-F1 0.847. Both errors were interpretable: *Chaat Street Food* (Cuisine) predicted Festivals, *Varanasi Ghats* (Landscape) predicted Festivals.

**K-Means** (k=6, no labels used for fitting): ARI 0.30, NMI 0.47, purity 0.62. Some clusters are clean (a Cuisine cluster 0.86 pure, a Music & Dance cluster 0.89, a silk-weaving cluster 1.0). Others group by shared descriptive words (pattern, festival, community, women) across categories.

## Objective 3: ranking quality and failures

Relevance = a result is relevant if its category is one of the query's relevant categories. Metrics: P@5, MRR, nDCG@10; a document with zero term overlap is never counted as retrieved. A grid of 8 configurations (n-grams x stemming x title in index) was scored; the dev-selected one is unigrams + stemming + title.

| Split | P@5 | MRR | nDCG@10 |
|---|---|---|---|
| Dev (18 queries) | 0.69 | 1.00 | 0.57 |
| Test (12 queries) | 0.55 | 0.92 | 0.45 |
| Keyword-overlap baseline, test | 0.55 | 0.88 | 0.43 |

What this shows: **stemming is the only variation that clearly mattered** (test P@5 0.45 -> 0.55 with unigrams and title); bigrams changed nothing for retrieval. TF-IDF is only marginally better than naive keyword overlap on this tiny set, so claim little beyond "the first hit is usually right" (MRR) while the top 5 is often diluted.

Failure patterns (14 queries below P@5 0.8; full list on `/evaluation` and in `results.json -> retrieval -> failures`):

1. **Vocabulary mismatch / no evidence:** "birdwatching" and "inlay" never occur (the corpus says "migratory birds", "inlaid"), so P@5 is 0.0 and 0.4 and few or no results are returned. "memorial of a massacre" matches a single document.
2. **Generic words spread across categories:** "community", "royal", "folk", "women", "street", "high" and "flower" appear in several categories, so wrong-category documents match on them (e.g. "folk painting" returns three Music & Dance folk dances).
3. **Category-level judgments undercount valid answers:** "festival of lights and sweets" returns Bengali Sweets, a sensible hit but Cuisine, not Festivals.
4. **Semantic gaps:** "what to eat in the south" needs the idea of food plus region; only the word "south" overlaps (Carnatic Music).

The interface handles case 1 explicitly: it warns about out-of-vocabulary terms, flags low-confidence results, and returns fewer than k results rather than padding with zero-score noise.

## Documented limitations

- The corpus is a **small starter set of short original descriptions** written for this project, not scraped text. Replace `data/heritage_corpus.csv` (same columns) with your selected sources and re-run; expect different numbers.
- ~69 documents and 12 test queries give wide uncertainty. Hold-out accuracy rests on 14 documents.
- Queries were written by the project author, who knew the corpus; real users will do worse.
- Lexical matching only: no synonyms or semantics. Embeddings or query expansion are the natural next step.
- Category is a coarse proxy for relevance and cannot judge individual documents.
- LLM summaries are optional, unverified, and constrained only by the prompt.
