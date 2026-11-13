"""A small, readable byte-pair encoding implementation.

Training follows the original recipe: pre-tokenise on whitespace and word
boundaries, start from the 256 byte values, and repeatedly merge the most
frequent adjacent pair. Ties are broken by the pair value, so the same corpus
and vocabulary size always produce the same tokenizer.

Pre-tokenisation keeps whitespace runs as their own units, which is what makes
``decode(encode(text)) == text`` hold for any Unicode input.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re

BASE_VOCAB = 256
_PRETOKEN = re.compile(r"\s+|\S+")


def pretokenize(text):
    """Split text into words and whitespace runs."""
    return _PRETOKEN.findall(text)


def _pairs(ids):
    return [(ids[index], ids[index + 1]) for index in range(len(ids) - 1)]


def _merge(ids, pair, new_id):
    merged = []
    index = 0
    while index < len(ids):
        if index < len(ids) - 1 and ids[index] == pair[0] and ids[index + 1] == pair[1]:
            merged.append(new_id)
            index += 2
        else:
            merged.append(ids[index])
            index += 1
    return merged


class Tokenizer:
    """A byte-level BPE tokenizer defined by an ordered list of merges."""

    def __init__(self, merges=(), special_tokens=()):
        self.merges = [(int(a), int(b), int(new)) for a, b, new in merges]
        self.ranks = {(a, b): rank for rank, (a, b, _new) in enumerate(self.merges)}
        self.special_tokens = tuple(special_tokens)
        self.vocab = {index: bytes([index]) for index in range(BASE_VOCAB)}
        for left, right, new_id in self.merges:
            self.vocab[new_id] = self.vocab[left] + self.vocab[right]

    def __len__(self):
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _encode_unit(self, unit):
        ids = list(unit.encode("utf-8"))
        while len(ids) >= 2:
            best_rank = None
            best_index = None
            for index in range(len(ids) - 1):
                rank = self.ranks.get((ids[index], ids[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, index
            if best_index is None:
                break
            ids[best_index : best_index + 2] = [self.merges[best_rank][2]]
        return ids

    def encode(self, text):
        ids = []
        for unit in pretokenize(text):
            ids.extend(self._encode_unit(unit))
        return ids

    def decode(self, ids):
        pieces = []
        for token_id in ids:
            piece = self.vocab.get(int(token_id))
            if piece is None:
                raise KeyError("unknown token id {}".format(token_id))
            pieces.append(piece)
        return b"".join(pieces).decode("utf-8", "replace")

    def piece(self, token_id):
        return self.vocab[int(token_id)]

    def segmentation(self, text):
        return [self.vocab[token_id] for token_id in self.encode(text)]

    def display(self, text, separator="|"):
        """Render a tokenisation with whitespace made visible."""
        rendered = []
        for piece in self.segmentation(text):
            rendered.append(piece.decode("utf-8", "replace").replace("\n", "\\n").replace(" ", "\u2423"))
        return separator.join(rendered)

    def merge_table(self, limit=None):
        rows = []
        for rank, (left, right, new_id) in enumerate(self.merges[: limit or len(self.merges)]):
            rows.append(
                {
                    "rank": rank,
                    "left": self.vocab[left].decode("utf-8", "replace"),
                    "right": self.vocab[right].decode("utf-8", "replace"),
                    "merged": self.vocab[new_id].decode("utf-8", "replace"),
                    "id": new_id,
                }
            )
        return rows

    def save(self, path):
        payload = {"version": 1, "merges": [list(merge) for merge in self.merges], "special_tokens": list(self.special_tokens)}
        pathlib.Path(path).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls(merges=payload.get("merges", ()), special_tokens=payload.get("special_tokens", ()))


def train(text, vocab_size=512, min_frequency=2, max_rounds=None):
    """Learn merges from ``text`` until ``vocab_size`` tokens or no pair repeats."""
    if vocab_size <= BASE_VOCAB:
        raise ValueError("vocab_size must be greater than {}".format(BASE_VOCAB))
    counts = collections.Counter(pretokenize(text))
    if not counts:
        raise ValueError("cannot train on empty text")
    words = {word: list(word.encode("utf-8")) for word in counts}
    stats = collections.Counter()
    where = collections.defaultdict(set)
    for word, ids in words.items():
        frequency = counts[word]
        for pair in _pairs(ids):
            stats[pair] += frequency
            where[pair].add(word)

    merges = []
    next_id = BASE_VOCAB
    budget = vocab_size - BASE_VOCAB if max_rounds is None else min(vocab_size - BASE_VOCAB, max_rounds)
    while len(merges) < budget and stats:
        pair, frequency = max(stats.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
        if frequency < min_frequency:
            break
        merges.append((pair[0], pair[1], next_id))
        for word in list(where.pop(pair, ())):
            ids = words[word]
            weight = counts[word]
            old_pairs = _pairs(ids)
            new_ids = _merge(ids, pair, next_id)
            if new_ids == ids:
                continue
            for old in old_pairs:
                stats[old] -= weight
                if stats[old] <= 0:
                    del stats[old]
                where[old].discard(word)
            for new_pair in _pairs(new_ids):
                stats[new_pair] += weight
                where[new_pair].add(word)
            words[word] = new_ids
        next_id += 1
    return Tokenizer(merges)


def compression_report(tokenizer, text):
    """Compare byte, character and token counts for a piece of text."""
    encoded = tokenizer.encode(text)
    characters = len(text)
    utf8_bytes = len(text.encode("utf-8"))
    return {
        "characters": characters,
        "utf8_bytes": utf8_bytes,
        "tokens": len(encoded),
        "vocab_size": tokenizer.vocab_size,
        "chars_per_token": round(characters / len(encoded), 3) if encoded else 0.0,
        "bytes_per_token": round(utf8_bytes / len(encoded), 3) if encoded else 0.0,
        "tokens_per_byte": round(len(encoded) / utf8_bytes, 3) if utf8_bytes else 0.0,
    }
