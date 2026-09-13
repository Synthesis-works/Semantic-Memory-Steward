"""Preprocessing layer that keeps embedding inputs inside Titan V2's hard caps.

Live-probed contract (amazon.titan-embed-text-v2:0, 2026-09-13) — BOTH limits
are enforced by the service, whichever trips first:

    - inputText maxLength is 50,000 CHARACTERS, enforced exactly:
      50,000 chars -> accepted (1024-dim output)
      50,001 chars -> ValidationException
      "Malformed input request: expected maxLength: 50000, actual: 50001, ..."
    - Max input TOKENS is 8,192, enforced exactly:
      40,051-char markdown (a real doc) -> rejected:
      "Too many input tokens. Max input tokens: 8192, request input token
       count: 10564". A char-dense probe repeated phrase embedded fine at
       45,000 chars (~3.9 chars/token), proving chars alone are NOT a safe
       proxy for the token limit.

Therefore the budget below enforces BOTH axes. estimate_tokens() is a
deterministic CONSERVATIVE over-estimate (latin ~3 chars/token, CJK/Kana/
Hangul 1 token/char), so an input that passes the estimator is very unlikely
to exceed the real 8,192-token limit.

Strategy (deterministic, content-preserving):
    1. Under every budget -> exact passthrough (small docs unchanged).
    2. Over either budget -> block-split, then global de-duplication keyed on
       a whitespace/digit-normalized block identity, keeping the FIRST
       occurrence's original text for every distinct block. Repeated padding
       (dated counters, log heartbeats, duplicate rows) collapses away while
       all distinct sections/messages survive.
    3. Still over -> head + tail reduction at block boundaries (title/early
       sections retained, late sections appended), with a marker separator,
       deterministically 65/35 weighted on BOTH the char axis and the token
       axis.
Returns a PreparedEmbeddingText record describing what changed; empty or
whitespace-only inputs pass straight through untouched.
"""
import math
import os
import re
from dataclasses import dataclass

# Hard service limits discovered by live probe.
HARD_INPUT_CAP_CHARS = 50_000
HARD_INPUT_CAP_TOKENS = 8_192
# Working budgets: 90% / 97.6% of the hard caps, leaving margin for
# tokenizer/estimator drift. Enforced together; either trip triggers reduction.
DEFAULT_MAX_INPUT_CHARS = 45_000
DEFAULT_MAX_INPUT_TOKENS = 8_000

_SEPARATOR = "\n[embedded from a reduced input; original content preserved in memory]\n"
_HEAD_WEIGHT = 0.65

_CJK_RE = re.compile(
    "[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff66-\uff9f\u3040-\u30ff]"
)


@dataclass
class PreparedEmbeddingText:
    text: str
    truncated: bool
    method: str          # "passthrough" | "collapse" | "collapse+headtail"
    chars_before: int
    chars_after: int
    original_tokens: int
    used_tokens: int
    collapsed_blocks: int

    @property
    def estimated_tokens(self) -> int:
        return self.original_tokens


def _max_chars() -> int:
    raw = os.getenv("SMS_EMBED_MAX_CHARS", "")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_MAX_INPUT_CHARS


def _max_tokens() -> int:
    raw = os.getenv("SMS_EMBED_MAX_TOKENS", "")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_MAX_INPUT_TOKENS


def estimate_tokens(text: str) -> int:
    """Deterministic CONSERVATIVE upper-ish bound used for enforcement.

    Models SentencePiece-like behavior on embedding content: CJK ideographs,
    kana, hangul and CJK punctuation cost ~1 token per character; latin
    takes ~3 chars per token; every whitespace-delimited run costs at least
    one token. Over-counting is safe (reduction happens earlier); a real
    8,192-token rejection from under-counting is what we are avoiding.
    """
    tokens = 0
    for word in re.findall(r"\S+", text):
        if not word:
            continue
        cjk = _CJK_RE.subn("", word)[1]
        latin = len(word) - cjk
        tokens += cjk + (max(1, math.ceil(latin / 3.0)) if latin else 0)
    return tokens


def _normalize_key(block: str) -> str:
    """Identity for de-duplication: digits folded to a constant, runs of
    whitespace collapsed. Keeps case and punctuation otherwise."""
    key = re.sub(r"\d", "0", block.strip())
    key = re.sub(r"\s+", " ", key)
    return key.strip()


def _blockize(text: str):
    """Split on blank lines (paragraph/section granularity). Documents without
    paragraph breaks (extracted PDFs, CSV rows, log dumps) collapse to their
    existing line blocks when newline-dense, else to a single block."""
    if re.search(r"\r?\n\s*\r?\n", text):
        blocks = [b.strip() for b in re.split(r"\r?\n\s*\r?\n", text)]
        return [b for b in blocks if b]
    blocks = [b.strip() for b in text.splitlines() if b.strip()]
    return blocks or ([text.strip()] if text.strip() else [])


def _dedup(blocks):
    """Global de-duplication by normalized identity; first occurrence wins."""
    kept, seen, collapsed = [], set(), 0
    for block in blocks:
        key = _normalize_key(block)
        if not key:
            continue
        if key in seen:
            collapsed += 1
            continue
        seen.add(key)
        kept.append(block)
    return kept, collapsed


def _block_costs(blocks, start, stop):
    """(chars, est_tokens) consumed by blocks[start:stop] joined with \\n\\n."""
    chars = sum(len(b) + 2 for b in blocks[start:stop])
    tokens = sum(estimate_tokens(b) for b in blocks[start:stop])
    return chars, tokens


def _head_tail(blocks, max_chars, max_tokens):
    """Head+tail at block boundaries; deterministic 65/35 split respecting
    BOTH the char budget and the token budget."""
    head_chars_guard = int(max_chars * _HEAD_WEIGHT)
    head_tokens_guard = int(max_tokens * _HEAD_WEIGHT)

    head = []
    head_chars = head_tokens = 0
    for i, block in enumerate(blocks):
        if head and (head_chars + len(block) + 2 > head_chars_guard or
                     head_tokens + estimate_tokens(block) > head_tokens_guard):
            break
        head.append(block)
        head_chars += len(block) + 2
        head_tokens += estimate_tokens(block)

    tail_chars_budget = max_chars - len(_SEPARATOR) - head_chars
    tail_tokens_budget = max_tokens - estimate_tokens(_SEPARATOR) - head_tokens

    tail = []
    tail_chars = tail_tokens = 0
    for block in reversed(blocks[len(head):]):
        size_c = len(block) + 2
        size_t = estimate_tokens(block)
        if tail and (tail_chars + size_c > tail_chars_budget or
                     tail_tokens + size_t > tail_tokens_budget):
            break
        tail.insert(0, block)
        tail_chars += size_c
        tail_tokens += size_t

    def join():
        return "\n\n".join(head) + _SEPARATOR + "\n\n".join(tail)

    text = join()

    # Guarantee both budgets WITHOUT slicing away the true ending: drop the
    # earliest tail blocks first, keeping the document's final block.
    while (tail and
           (len(text) > max_chars or estimate_tokens(text) > max_tokens)):
        tail.pop(0)
        text = join()

    if estimate_tokens(text) > max_tokens and head:
        # Last resort (e.g. one pathological single block): shrink from the
        # front so the title/head survives, never mid-content. Binary-search
        # the longest prefix that fits the token budget (and the char budget).
        sep_tokens = estimate_tokens(_SEPARATOR)
        target = max(0, max_tokens - sep_tokens)
        hi = min(len(head[0]), max(0, max_chars - len(_SEPARATOR)))
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if estimate_tokens(head[0][:mid]) <= target:
                lo = mid
            else:
                hi = mid - 1
        text = (head[0][:lo] if lo else "") + _SEPARATOR

    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def prepare_embedding_text(text: str, max_chars: int = None, max_tokens: int = None):
    """Reduce *text* to a deterministic embedding input at or under budget.

    An input that satisfies BOTH the char budget and the (conservative) token
    budget is returned verbatim (byte-for-byte identical string); anything
    over either budget is reduced. Empty/whitespace-only inputs pass through.
    """
    chars_budget = (max_chars or _max_chars())
    tokens_budget = (max_tokens or _max_tokens())
    tokens = estimate_tokens(text)

    if (not text.strip() or
            (len(text) <= chars_budget and tokens <= tokens_budget)):
        return PreparedEmbeddingText(
            text=text, truncated=False, method="passthrough",
            chars_before=len(text), chars_after=len(text),
            original_tokens=tokens, used_tokens=tokens, collapsed_blocks=0,
        )

    blocks = _blockize(text)
    kept, collapsed = _dedup(blocks)
    reduced = "\n\n".join(kept)
    reduced_tokens = estimate_tokens(reduced)

    if len(reduced) <= chars_budget and reduced_tokens <= tokens_budget:
        return PreparedEmbeddingText(
            text=reduced, truncated=True, method="collapse",
            chars_before=len(text), chars_after=len(reduced),
            original_tokens=tokens, used_tokens=reduced_tokens,
            collapsed_blocks=collapsed,
        )

    cut = _head_tail(kept, chars_budget, tokens_budget)
    used = estimate_tokens(cut)
    return PreparedEmbeddingText(
        text=cut, truncated=True, method="collapse+headtail",
        chars_before=len(text), chars_after=len(cut),
        original_tokens=tokens, used_tokens=used, collapsed_blocks=collapsed,
    )