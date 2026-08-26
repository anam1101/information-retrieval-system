# Information Retrieval System

A search engine built from scratch: a **positional inverted index** for exact phrase queries, and **five TF weighting schemes** compared side by side for ranked retrieval — no search library doing the work underneath.

## What it does

**Phrase queries.** A plain inverted index tells you a document contains "quick", "brown", and "fox". It can't tell you they appear *consecutively*. This index stores every token's positions per document, so `"quick brown fox"` matches only documents where those words are genuinely adjacent — verified by checking position offsets, not just co-occurrence.

**Ranked retrieval with five weighting schemes.** The same query is scored five ways so the schemes can be compared directly:

| Scheme | Weight |
|---|---|
| Binary | 1 if present |
| Raw count | `tf` |
| Normalized | `tf / total terms in document` |
| Log | `log(1 + tf)` |
| Double normalization | `0.5 + 0.5 × (tf / max tf in document)` |

Documents and queries are vectorised into the same space and ranked by **cosine similarity**. Term–document matrices are held as **sparse** matrices, so vocabulary size doesn't blow up memory.

## Running it

```bash
pip install -r requirements.txt
python ir_system.py
```

NLTK corpora download automatically on first run only. An interactive menu offers phrase search or ranked retrieval.

## Corpus

`data/` contains a 25-document sample so the project runs immediately after cloning. Point `DATA_DIR` at any folder of `.txt` files to use your own corpus — the original run indexed 249 documents and a ~47,000-term vocabulary.

## Tests

```bash
python -m pytest tests/ -q
```

9 tests covering preprocessing, positional index construction, and phrase matching — including that word *order* is respected (`"fox brown"` must not match `"brown fox"`) and that adjacency is genuinely enforced across documents.

## Built with

Python 3 · NLTK (tokenisation, stopwords) · scikit-learn (`CountVectorizer`, cosine similarity) · SciPy sparse matrices · NumPy
