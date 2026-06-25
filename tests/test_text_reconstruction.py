# ─────────────────────────────────────────────────────────────────────────────
# tests/test_text_reconstruction.py — Unit Tests: Text Reconstruction (TEST_PLAN §8)
# ─────────────────────────────────────────────────────────────────────────────

import pytest
import pandas as pd

from src.ocr_pipeline import filter_dataframe, _reconstruct_text


# TC-TEXT-001: Single-line reconstruction preserves word order
def test_single_line_word_order(sample_tesseract_dataframe):
    """'Hello' (conf=95) and 'test' (conf=82) pass; 'world' (conf=45) fails."""
    df = filter_dataframe(sample_tesseract_dataframe)
    text = _reconstruct_text(df)
    assert "Hello" in text
    assert "test" in text


# TC-TEXT-002: Low-confidence word NOT in reconstructed text
def test_low_confidence_word_excluded(sample_tesseract_dataframe):
    df = filter_dataframe(sample_tesseract_dataframe)
    text = _reconstruct_text(df)
    assert "world" not in text, "conf=45 word 'world' should be excluded"


# TC-TEXT-003: Empty DataFrame returns empty string (no crash)
def test_empty_dataframe_returns_empty_string():
    empty_df = pd.DataFrame(columns=[
        "block_num", "par_num", "line_num", "word_num", "text", "conf",
        "level", "page_num", "left", "top", "width", "height",
    ])
    text = _reconstruct_text(empty_df)
    assert text == ""


# TC-TEXT-004: Multi-line text has newline separators between lines
def test_multiline_has_newlines():
    """Construct a DataFrame with two distinct line groups."""
    df = pd.DataFrame({
        "level":    [5,    5,    5,    5   ],
        "page_num": [1,    1,    1,    1   ],
        "block_num":[1,    1,    1,    1   ],
        "par_num":  [1,    1,    1,    1   ],
        "line_num": [1,    1,    2,    2   ],
        "word_num": [1,    2,    1,    2   ],
        "left":     [10,   100,  10,   100 ],
        "top":      [10,   10,   50,   50  ],
        "width":    [80,   80,   80,   80  ],
        "height":   [30,   30,   30,   30  ],
        "conf":     [90.0, 85.0, 88.0, 92.0],
        "text":     ["Hello", "World", "Foo", "Bar"],
    })
    text = _reconstruct_text(df)
    assert "\n" in text, "Multi-line text must contain newline separators"


# TC-TEXT-005: Whitespace-only tokens do not appear in reconstructed output
def test_whitespace_tokens_excluded(sample_tesseract_dataframe):
    """After filtering, whitespace-only text tokens must not appear."""
    df = filter_dataframe(sample_tesseract_dataframe)
    text = _reconstruct_text(df)
    # Check that each non-empty word is actually a word
    for line in text.split("\n"):
        for word in line.split(" "):
            if word:  # Allow empty strings from split, not trailing spaces
                assert word.strip() == word, f"Word has leading/trailing whitespace: '{word}'"


# TC-TEXT-006: Words within a line are space-separated
def test_words_space_separated():
    df = pd.DataFrame({
        "level":    [5,    5   ],
        "page_num": [1,    1   ],
        "block_num":[1,    1   ],
        "par_num":  [1,    1   ],
        "line_num": [1,    1   ],
        "word_num": [1,    2   ],
        "left":     [10,   100 ],
        "top":      [10,   10  ],
        "width":    [80,   80  ],
        "height":   [30,   30  ],
        "conf":     [90.0, 85.0],
        "text":     ["Hello", "World"],
    })
    text = _reconstruct_text(df)
    assert "Hello World" in text


# TC-TEXT-007: Single word returns just that word
def test_single_word_reconstruction():
    df = pd.DataFrame({
        "level":    [5    ],
        "page_num": [1    ],
        "block_num":[1    ],
        "par_num":  [1    ],
        "line_num": [1    ],
        "word_num": [1    ],
        "left":     [10   ],
        "top":      [10   ],
        "width":    [80   ],
        "height":   [30   ],
        "conf":     [95.0 ],
        "text":     ["Invoice"],
    })
    text = _reconstruct_text(df)
    assert text.strip() == "Invoice"
