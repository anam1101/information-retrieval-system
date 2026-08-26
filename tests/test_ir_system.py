"""Tests for the positional-index IR system."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir_system import preprocess, build_positional_index, phrase_query, load_documents

DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick brown cat naps in the sun.",
    "Dogs and foxes are not the same animal.",
]


def test_preprocess_lowercases_and_strips_stopwords():
    tokens = preprocess("The Quick Brown Fox!")
    assert "quick" in tokens
    assert "the" not in tokens      # stopword
    assert "!" not in tokens        # punctuation
    assert all(t == t.lower() for t in tokens)


def test_preprocess_empty_string():
    assert preprocess("") == []


def test_positional_index_records_positions():
    index = build_positional_index(DOCS)
    # "quick" appears in docs 0 and 1
    assert set(index["quick"].keys()) == {0, 1}
    # positions are recorded as integer offsets
    assert all(isinstance(p, int) for p in index["quick"][0])


def test_phrase_query_matches_adjacent_terms():
    index = build_positional_index(DOCS)
    assert 0 in phrase_query(index, DOCS, "quick brown fox")


def test_phrase_query_respects_word_order():
    index = build_positional_index(DOCS)
    # "fox brown" is not an adjacent phrase in that order
    assert phrase_query(index, DOCS, "fox brown") == []


def test_phrase_query_distinguishes_documents():
    index = build_positional_index(DOCS)
    assert phrase_query(index, DOCS, "brown cat") == [1]


def test_phrase_query_empty_input():
    index = build_positional_index(DOCS)
    assert phrase_query(index, DOCS, "") == []


def test_phrase_query_unknown_term():
    index = build_positional_index(DOCS)
    assert phrase_query(index, DOCS, "helicopter") == []


def test_load_documents_reads_corpus(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "b.txt").write_text("")          # empty -> skipped
    (tmp_path / "c.md").write_text("ignored")    # wrong extension
    docs, names = load_documents(str(tmp_path))
    assert names == ["a.txt"]
    assert docs == ["hello world"]
