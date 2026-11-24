"""Flask interface: ranked heritage search with matching-term evidence.

Run:  python app.py      then open http://127.0.0.1:5000
Optional LLM summary:  export ANTHROPIC_API_KEY=...   (falls back to extractive)
"""
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.llm import llm_available, summarize
from src.preprocess import load_corpus
from src.search import HeritageIndex, IndexConfig

ROOT = Path(__file__).parent
LOW_CONFIDENCE = 0.15  # top cosine score below this is flagged to the user

app = Flask(__name__)

# Configuration chosen on the DEV queries in run_experiments.py (see outputs/results.json).
_df, _report = load_corpus(ROOT / "data" / "heritage_corpus.csv", stem=True)
INDEX = HeritageIndex(IndexConfig(ngram_range=(1, 1), stem=True, use_title=True)).fit(_df)
CATEGORIES = sorted(_df["category"].unique())


CAT_COLORS = {
    "Monuments & Architecture": "#B4471F",
    "Festivals & Rituals": "#C28A12",
    "Crafts & Textiles": "#8A3D7A",
    "Music & Dance": "#2B6CB0",
    "Cuisine & Foodways": "#C0392B",
    "Natural & Sacred Landscapes": "#2F7D4F",
}
CAT_ICONS = {
    "Monuments & Architecture": "🏛️",
    "Festivals & Rituals": "🪔",
    "Crafts & Textiles": "🧵",
    "Music & Dance": "🎶",
    "Cuisine & Foodways": "🍛",
    "Natural & Sacred Landscapes": "🌿",
}


@app.context_processor
def inject_style_helpers():
    return {"cat_color": lambda c: CAT_COLORS.get(c, "#7a6a5a"),
            "cat_icon": lambda c: CAT_ICONS.get(c, "📜")}


def _results_path():
    return ROOT / "outputs" / "results.json"


def run_search(q: str, k: int, category: str | None, want_summary: bool):
    q = (q or "").strip()
    if not q:
        return {"query": q, "results": [], "warnings": [], "summary": None}
    results = INDEX.search(q, k=k, category=category or None)
    warnings = []
    oov = INDEX.oov_terms(q)
    if oov:
        warnings.append("Not found anywhere in the collection: " + ", ".join(oov) +
                        ". Try a different wording.")
    if not results:
        warnings.append("No document shares any term with this search, so nothing is ranked.")
    elif results[0]["score"] < LOW_CONFIDENCE:
        warnings.append("Low confidence: even the best match overlaps with your search on only "
                        "a few generic words.")
    elif len(results) < k:
        warnings.append(f"Only {len(results)} documents share terms with your search.")
    summary = None
    if want_summary and results:
        text = summarize(q, results)
        if text:
            summary = {"kind": "llm", "text": text}
        else:
            summary = {"kind": "extractive",
                       "text": " ".join(r["snippet"] for r in results[:3])}
    return {"query": q, "results": results, "warnings": warnings, "summary": summary}


@app.route("/")
def home():
    q = request.args.get("q", "")
    k = min(max(request.args.get("k", 5, type=int), 1), 20)
    category = request.args.get("category", "")
    want_summary = request.args.get("summary") == "1"
    data = run_search(q, k, category, want_summary)
    return render_template("index.html", data=data, k=k, category=category,
                           categories=CATEGORIES, want_summary=want_summary,
                           llm=llm_available(), searched=bool(q.strip()),
                           clusters=INDEX.cluster_terms)


@app.route("/api/search")
def api_search():
    k = min(max(request.args.get("k", 5, type=int), 1), 20)
    data = run_search(request.args.get("q", ""), k, request.args.get("category"),
                      request.args.get("summary") == "1")
    for r in data["results"]:
        r.pop("text", None)
    return jsonify(data)


@app.route("/evaluation")
def evaluation():
    path = _results_path()
    results = json.loads(path.read_text()) if path.exists() else None
    return render_template("evaluation.html", r=results)


@app.route("/health")
def health():
    return jsonify(status="ok", documents=len(INDEX.df), terms=len(INDEX.terms),
                   llm_summaries=llm_available())


if __name__ == "__main__":
    app.run(debug=False, port=5000)
