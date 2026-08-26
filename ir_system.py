import os
import string
import math
import numpy as np
import nltk
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer



def ensure_nltk_resources():
    """Download required NLTK corpora once, instead of on every run."""
    required = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ]
    for package, path in required:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

DATA_DIR = "data"
STOPWORDS = set(stopwords.words("english"))
PUNCTUATION = set(string.punctuation)


def load_documents(data_dir):
    docs = []
    doc_names = []
    errors = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".txt"):
            try:
                with open(
                    os.path.join(data_dir, fname),
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ) as f:
                    content = f.read()
                    if not content.strip():
                        errors.append(f"{fname}: empty file")
                        continue
                    docs.append(content)
                    doc_names.append(fname)
            except Exception as e:
                errors.append(f"{fname}: {str(e)}")
    if errors:
        print("\n[Warning] Some files could not be loaded or were empty:")
        for err in errors:
            print("-", err)
    return docs, doc_names


def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and t not in PUNCTUATION]
    tokens = [t for t in tokens if t.strip()]
    return tokens


def build_positional_index(docs):
    index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in enumerate(docs):
        tokens = preprocess(text)
        for pos, token in enumerate(tokens):
            index[token][doc_id].append(pos)
    return index


def phrase_query(index, docs, phrase):
    phrase_tokens = preprocess(phrase)
    if not phrase_tokens:
        return []
    postings = index.get(phrase_tokens[0], {})
    result_docs = []
    for doc_id in postings:
        positions = postings[doc_id]
        for pos in positions:
            match = True
            for offset, token in enumerate(phrase_tokens[1:], 1):
                if doc_id not in index.get(token, {}):
                    match = False
                    break
                if (pos + offset) not in index[token][doc_id]:
                    match = False
                    break
            if match:
                result_docs.append(doc_id)
                break
    return result_docs


def main():
    print("Loading documents...")
    docs, doc_names = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents.")

    print("Preprocessing all documents once ...")
    preprocessed_docs = [preprocess(doc) for doc in docs]
    joined_docs = [" ".join(tokens) for tokens in preprocessed_docs]
    print("Building vocabulary ...")
    vocab = sorted(set(token for tokens in preprocessed_docs for token in tokens))
    print(f"Vocabulary size: {len(vocab)}")

    print("Building positional index ...")
    pos_index = build_positional_index(docs)
    print("Positional index built.")

    # Precompute TF/TF-IDF matrices for all schemes using sklearn (sparse)
    schemes = ["binary", "raw", "normalized", "log", "double"]
    print("Precomputing document-term matrices using sklearn ...")
    count_vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False)
    X_counts = count_vectorizer.fit_transform(joined_docs)
    X_binary = (X_counts > 0).astype(int)
    X_normalized = X_counts.multiply(1 / X_counts.sum(axis=1))
    X_log = X_counts.copy().astype(float)
    X_log.data = np.log1p(X_log.data)
    max_tf = X_counts.max(axis=1).toarray().flatten()
    X_double = X_counts.copy().astype(float)
    for i in range(X_double.shape[0]):
        if max_tf[i] > 0:
            X_double.data[X_double.indptr[i] : X_double.indptr[i + 1]] = 0.5 + 0.5 * (
                X_double.data[X_double.indptr[i] : X_double.indptr[i + 1]] / max_tf[i]
            )
    tfidf_matrices = {
        "binary": X_binary,
        "raw": X_counts,
        "normalized": X_normalized,
        "log": X_log,
        "double": X_double,
    }
    print("All document-term matrices precomputed (sparse).")

    def rank_documents_sparse(query, matrix, scheme, doc_names, doc_indices=None):
        query_tokens = preprocess(query)
        query_str = " ".join(query_tokens)
        q_counts = count_vectorizer.transform([query_str])
        if scheme == "binary":
            q_vec = (q_counts > 0).astype(int)
        elif scheme == "raw":
            q_vec = q_counts
        elif scheme == "normalized":
            q_vec = q_counts.multiply(1 / q_counts.sum(axis=1))
        elif scheme == "log":
            q_vec = q_counts.copy().astype(float)
            q_vec.data = np.log1p(q_vec.data)
        elif scheme == "double":
            max_tf = q_counts.max(axis=1).toarray().flatten()
            q_vec = q_counts.copy().astype(float)
            for i in range(q_vec.shape[0]):
                if max_tf[i] > 0:
                    q_vec.data[q_vec.indptr[i] : q_vec.indptr[i + 1]] = 0.5 + 0.5 * (
                        q_vec.data[q_vec.indptr[i] : q_vec.indptr[i + 1]] / max_tf[i]
                    )
        else:
            q_vec = q_counts
        from sklearn.metrics.pairwise import cosine_similarity as cs

        if doc_indices is not None:
            matrix = matrix.tocsr()[doc_indices]
            names = [doc_names[i] for i in doc_indices]
        else:
            names = doc_names
        sims = cs(q_vec, matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:5]
        return [(names[i], sims[i]) for i in top_idx if sims[i] > 0]

    while True:
        print("\nOptions:")
        print("1. Phrase Query (up to 5 terms)")
        print("2. Document Ranking (TF-IDF, 5 schemes)")
        print("3. Exit")
        choice = input("Select option: ")
        if choice == "1":
            phrase = input("Enter phrase query: ")
            if not phrase.strip():
                print("Empty query. Try again.")
                continue
            result_docs = phrase_query(pos_index, docs, phrase)
            if result_docs:
                print("Documents matching phrase:")
                for doc_id in result_docs:
                    print(f"- {doc_names[doc_id]}")
            else:
                print("No documents matched the phrase.")
        elif choice == "2":
            query = input("Enter search query: ")
            if not query.strip():
                print("Empty query. Try again.")
                continue
            phrase = query
            result_docs = phrase_query(pos_index, docs, phrase)
            if not result_docs:
                print("No documents matched the phrase. Ranking skipped.")
                continue
            doc_indices = result_docs
            for scheme in schemes:
                print(f"\nRanking documents for scheme: {scheme} ...")
                matrix = tfidf_matrices[scheme]
                top_docs = rank_documents_sparse(
                    query, matrix, scheme, doc_names, doc_indices
                )
                if not top_docs:
                    print("No matching documents found for this scheme.")
                for name, score in top_docs:
                    print(f"{name} (score: {score:.4f})")
        elif choice == "3":
            print("Exiting.")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
