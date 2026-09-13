"""Tests for the embedding-input preprocessing layer.

The layer exists because amazon.titan-embed-text-v2:0 enforces BOTH limits
(live-probed 2026-09-13): inputText maxLength = 50,000 characters AND max
input tokens = 8,192. Budget is enforced on both axes; the token estimator is
deliberately conservative (latin ~3 chars/token, CJK 1 token/char).
"""
import os
from unittest.mock import patch

from sms_agent.embed_prep import (
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_INPUT_TOKENS,
    HARD_INPUT_CAP_CHARS,
    HARD_INPUT_CAP_TOKENS,
    estimate_tokens,
    prepare_embedding_text,
)


def test_under_limit_input_is_passthrough_byte_identical():
    text = "This is a financial report.\nClose the books quarterly."
    prep = prepare_embedding_text(text, max_chars=500)
    assert prep.text == text
    assert prep.truncated is False
    assert prep.method == "passthrough"
    assert prep.chars_before == prep.chars_after == len(text)


def test_exactly_at_limit_input_is_passthrough():
    text = "x" * 500
    prep = prepare_embedding_text(text, max_chars=500)
    assert prep.text == text
    assert prep.truncated is False
    assert prep.method == "passthrough"


def test_one_over_limit_is_preprocessed():
    text = "y" * 501
    prep = prepare_embedding_text(text, max_chars=500)
    assert prep.truncated is True
    assert prep.text != text
    assert len(prep.text) <= 500


def test_oversized_input_always_within_budget():
    text = ("Northwind checkout standby pool health checks drain operations "
            "ledger reconciliation. ") * 5000
    for max_chars in (120, 500, 5000, DEFAULT_MAX_INPUT_CHARS):
        prep = prepare_embedding_text(text, max_chars=max_chars)
        assert len(prep.text) <= max_chars, f"over budget {max_chars}"


def test_deterministic_output_across_calls():
    text = ("Section one content remains stable. " * 3000) + \
           ("Section two content stays deterministic. " * 3000)
    a = prepare_embedding_text(text, max_chars=5000)
    b = prepare_embedding_text(text, max_chars=5000)
    assert a.text == b.text
    assert a.method == b.method
    assert a.collapsed_blocks == b.collapsed_blocks


def test_head_tail_retains_title_middle_and_tail_for_unique_text():
    # All blocks unique (no de-duplication possible) -> head+tail fallback.
    blocks = [
        "Paragraph " + "p" * (i + 1) + " long unique body text." * 10
        for i in range(400)
    ]
    text = "\n\n".join(blocks)
    assert len(text) > 20000
    prep = prepare_embedding_text(text, max_chars=20000)
    assert prep.method == "collapse+headtail"
    assert prep.truncated is True
    assert len(prep.text) <= 20000
    # Head & tail content survive; the separator marker is present.
    assert prep.text.startswith(blocks[0][:40])
    assert prep.text.endswith(blocks[-1][-40:])
    assert "reduced input" in prep.text


def test_repeated_padding_collapses_keeps_distinct_sections():
    # Seed-style md: a few distinct sections repeated with counting markers.
    section_a = "\n\n## Services\n\nHealth checks gate every drain operation."
    section_b = "\n\n## Data\n\nOrders ledger with nightly reconciliation."
    text = "# Architecture Overview" + section_a + "\n\n_Dated entry 0._"
    for i in range(1, 300):
        text += (section_a if i % 2 == 0 else section_b) + \
                f"\n\n_Dated entry {i}._"
    assert len(text) > 20000
    prep = prepare_embedding_text(text, max_chars=20000)
    assert prep.method == "collapse"
    assert prep.truncated is True
    assert prep.collapsed_blocks > 0
    assert "# Architecture Overview" in prep.text
    assert "## Services" in prep.text
    assert "## Data" in prep.text
    assert len(prep.text) <= 20000


def test_single_block_oversized_head_preserved_within_budget():
    # PDF/csv-log style: text with no paragraph breaks is one giant block;
    # tail cannot survive alone, but the head must and length must fit.
    text = "".join(
        f"2025-09-01,travel,{120 + i},field visit {i + 1};"
        for i in range(4000)
    )
    assert len(text) > 100000
    prep = prepare_embedding_text(text, max_chars=20000)
    assert prep.truncated is True
    assert prep.text.startswith("2025-09-01,travel,120,field visit 1;")
    assert len(prep.text) <= 20000


def test_multi_block_oversized_keeps_both_ends():
    # Unique paragraphs (digit-free identifiers) -> head+tail preserves ends.
    blocks = [
        "Paragraph " + "q" * (i + 1) + " body text. " * 12
        for i in range(300)
    ]
    text = "\n\n".join(blocks)
    assert len(text) > 30000
    prep = prepare_embedding_text(text, max_chars=20000)
    assert prep.method == "collapse+headtail"
    assert prep.text.startswith("Paragraph q ")
    assert prep.text.endswith(blocks[-1][-40:].rstrip())
    assert len(prep.text) <= 20000


def test_utf16_decoded_content_is_treated_as_plain_str():
    text = "Alice, Bob, Charlie\r\n\u2014 em dash and CJK 中文重型文字"
    prep = prepare_embedding_text(text, max_chars=500)
    assert prep.text == text
    assert prep.method == "passthrough"


def test_empty_and_whitespace_only_pass_through():
    assert prepare_embedding_text("", max_chars=10).text == ""
    ws = "   \n \t \r\n  "
    prep = prepare_embedding_text(ws, max_chars=10)
    assert prep.text == ws
    assert prep.truncated is False


def test_bom_char_does_not_disturb_content():
    text = "\ufeffProject Alpha Plan V1\r\n"
    prep = prepare_embedding_text(text, max_chars=500)
    assert prep.text == text
    assert prep.truncated is False


def test_malformed_control_characters_do_not_crash():
    text = "\x00\x01\x02" + "A" * 30000 + "\x1f\x7f" + "B" * 30001
    prep = prepare_embedding_text(text, max_chars=30000)
    assert len(prep.text) <= 30000
    assert prep.truncated is True


def test_env_override_changes_budget():
    with patch.dict(os.environ, {"SMS_EMBED_MAX_CHARS": "777"}, clear=False):
        prep = prepare_embedding_text("x" * 778)
    assert prep.text != "x" * 778
    assert len(prep.text) <= 777


def test_estimate_tokens_is_deterministic_and_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") == estimate_tokens("hello world")
    assert estimate_tokens("a b c") <= estimate_tokens("aa bb cc")


def test_default_budget_is_safe_under_hard_cap():
    assert DEFAULT_MAX_INPUT_CHARS < HARD_INPUT_CAP_CHARS
    assert DEFAULT_MAX_INPUT_TOKENS < HARD_INPUT_CAP_TOKENS


def test_default_budget_boundary():
    text = "z" * (DEFAULT_MAX_INPUT_CHARS + 1)
    prep = prepare_embedding_text(text)
    assert prep.truncated is True
    assert len(prep.text) <= DEFAULT_MAX_INPUT_CHARS


def test_estimate_tokens_counts_cjk_densely():
    assert estimate_tokens("数据" * 10) == 20
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("a b") == 2
    assert estimate_tokens("héllo wörld") == estimate_tokens("hello world")


def test_cjk_dense_below_char_budget_is_still_reduced():
    # 8,100 chars < char budget (45,000) but > token budget -> must reduce.
    text = "重" * 8100
    assert len(text) <= DEFAULT_MAX_INPUT_CHARS
    prep = prepare_embedding_text(text)
    assert prep.truncated is True
    assert prep.method == "collapse+headtail"
    assert len(prep.text) <= DEFAULT_MAX_INPUT_CHARS
    assert estimate_tokens(prep.text) <= DEFAULT_MAX_INPUT_TOKENS


def test_exact_token_boundary_passes_through_one_word_over_reduces():
    # max_tokens=100 boundary: "w " * 100 is exactly at the token budget.
    at_limit = "w " * 100
    prep = prepare_embedding_text(at_limit, max_chars=100000, max_tokens=100)
    assert prep.text == at_limit
    assert prep.method == "passthrough"

    over = "w " * 101
    prep2 = prepare_embedding_text(over, max_chars=100000, max_tokens=100)
    assert prep2.truncated is True
    assert prep2.text != over
    assert estimate_tokens(prep2.text) <= 100
    assert prep2.used_tokens <= 100


def test_headtail_respects_token_budget_within_char_budget():
    # Unique CJK blocks (no digits -> normalize does not fuse them): chars fit
    # the char budget, tokens do not.
    blocks = [f"重" * 300 + f"节" * (i + 1) for i in range(30)]
    text = "\n\n".join(blocks)
    assert len(text) <= DEFAULT_MAX_INPUT_CHARS
    assert estimate_tokens(text) > DEFAULT_MAX_INPUT_TOKENS
    prep = prepare_embedding_text(text)
    assert prep.method == "collapse+headtail"
    assert estimate_tokens(prep.text) <= DEFAULT_MAX_INPUT_TOKENS
    assert len(prep.text) <= DEFAULT_MAX_INPUT_CHARS