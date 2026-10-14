"""Verify candidate BPE segmentation against public canonical GPT-2 assets.

Standard-library-only; fetches vocabulary and merge ranks, without model weights
or package installation. The two inputs each match one GPT-2 pretokenizer unit
(`` ?\\p{L}+``), so no third-party Unicode regex package is needed here.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = "https://huggingface.co/openai-community/gpt2/resolve/main/"


def fetch(name):
    with urllib.request.urlopen(BASE + name, timeout=60) as response:
        raw = response.read()
        return raw, {
            "url": BASE + name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "etag": response.headers.get("ETag"),
            "resolved_url": response.url,
        }


vocab_raw, vocab_metadata = fetch("vocab.json")
merges_raw, merges_metadata = fetch("merges.txt")
vocab = json.loads(vocab_raw)
merge_lines = merges_raw.decode("utf-8").splitlines()
assert merge_lines[0].startswith("#version:")
merge_ranks = {tuple(line.split()): rank for rank, line in enumerate(merge_lines[1:]) if line}

# GPT-2 byte-to-Unicode lookup: byte 32 (space) maps to U+0120 (G with dot).
byte_values = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
unicode_values = byte_values.copy()
extra = 0
for value in range(256):
    if value not in byte_values:
        byte_values.append(value)
        unicode_values.append(256 + extra)
        extra += 1
byte_encoder = dict(zip(byte_values, map(chr, unicode_values)))


def encode_candidate(text):
    assert text.startswith(" ") and text[1:].isascii() and text[1:].isalpha()
    pieces = tuple(byte_encoder[value] for value in text.encode("utf-8"))
    while len(pieces) > 1:
        pairs = set(zip(pieces, pieces[1:]))
        best = min(pairs, key=lambda pair: merge_ranks.get(pair, float("inf")))
        if best not in merge_ranks:
            break
        merged = []
        index = 0
        while index < len(pieces):
            if index + 1 < len(pieces) and pieces[index:index + 2] == best:
                merged.append(pieces[index] + pieces[index + 1])
                index += 2
            else:
                merged.append(pieces[index])
                index += 1
        pieces = tuple(merged)
    return list(pieces), [vocab[piece] for piece in pieces]


results = []
for text, expected_id in [(" Paris", 6342), (" London", 3576)]:
    pieces, ids = encode_candidate(text)
    assert ids == [expected_id], (text, pieces, ids)
    results.append({"text": text, "bpe_pieces": pieces, "token_ids": ids, "n_tokens": len(ids)})

report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
          "assets": {"vocab": vocab_metadata, "merges": merges_metadata},
          "method": "GPT-2 byte encoding followed by lowest-rank adjacent BPE merges; one pretokenizer word per candidate",
          "candidates": results}
output = Path(__file__).with_name("gpt2_candidate_tokenization.json")
output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(json.dumps(results, ensure_ascii=True))
print("Saved canonical BPE verification:", output)
