# bpelab

**Byte-pair encoding from scratch, small enough to read in one sitting.**

Tokenizers are the part of modern language models people skip: a hundred lines of merge logic that decide what the model can see. `bpelab` trains a real byte-level BPE tokenizer on your own text, shows you the merges it learned, and measures how much it actually compresses. No dependencies, no downloads, no model weights.

```bash
python -m bpelab train corpus.txt --vocab-size 400 --show-merges 5
```

`train` prints a compression report and the merge table it learned:

```text
{
  "characters": ...,
  "utf8_bytes": ...,
  "tokens": ...,
  "vocab_size": 400,
  "chars_per_token": ...,
  "bytes_per_token": ...,
  "tokens_per_byte": ...
}

first merges learned (most frequent first):
  #0            ' ' + ' '          -> '  '
  #1           '  ' + '  '         -> '    '
  #2            'e' + 'n'          -> 'en'
  #3            'i' + 'n'          -> 'in'
```

Those are the real first merges from a whitespace-heavy Markdown corpus: indentation gets merged before words do, because it genuinely is the most frequent adjacent pair. Train on ordinary prose and the first merges are letter pairs like `th`, `he`, `in`; train on Python source and the winners are `    `, `self`, `(`, `)`, ` = `.

The whole algorithm fits in your head: start from bytes, count adjacent pairs, merge the most frequent one, repeat. Each merge adds exactly one token to the vocabulary.

## Install

```bash
python -m pip install -e .
```

Python 3.9+, standard library only.

## What it does

1. **Pre-tokenises** on whitespace and word boundaries, keeping whitespace runs as their own units. That is what makes `decode(encode(text)) == text` true for any Unicode input, including emoji, tabs and trailing spaces.
2. **Trains** by counting adjacent pairs weighted by word frequency and merging the winner. Pair statistics are updated incrementally, so training stays fast even in pure Python.
3. **Encodes** by applying merges in rank order (the standard greedy algorithm), and **decodes** by concatenating the bytes each token stands for.
4. **Reports** tokens, bytes per token and compression, plus the merge table so you can see what the vocabulary is made of.

## Commands

```bash
# learn a tokenizer and save it
bpelab train corpus.txt --vocab-size 800 -o tokenizer.json

# inspect a segmentation (whitespace shown as U+2423)
bpelab encode tokenizer.json "the quick brown fox"
bpelab encode tokenizer.json "the quick brown fox" --ids
bpelab encode tokenizer.json "the quick brown fox" --pieces

# measure a tokenizer on a corpus
bpelab stats tokenizer.json corpus.txt

# how does compression scale with vocabulary size?
bpelab demo corpus.txt --sizes 300,500,1000,2000
```

`bpelab demo` prints a table comparing vocabulary sizes:

```text
 vocab     tokens  chars/token    tokens/byte
   300        ...          ...            ...
   500        ...          ...            ...
  1000        ...          ...            ...
```

Bigger vocabularies mean fewer tokens for the same text, at the cost of a larger embedding matrix. That trade-off is the whole reason tokenizer design is still interesting.

## Library use

```python
from bpelab import Tokenizer, train, compression_report

tokenizer = train(open("corpus.txt").read(), vocab_size=1000)
ids = tokenizer.encode("hello world")
assert tokenizer.decode(ids) == "hello world"
print(tokenizer.display("hello world"))     # h|ello|\u2423|world
tokenizer.save("tokenizer.json")
print(compression_report(tokenizer, "hello world"))
```

## What it deliberately does not do

- No NFKC normalisation or language-specific pre-tokenisation regex; `bpelab` splits on whitespace only.
- No special-token splicing into arbitrary text and no chat templates.
- No parallelism: this is a teaching implementation, not a training pipeline for a 100k vocabulary.

Those are exactly the parts worth adding once the hundred lines are clear.

## Tests

```bash
python -m pytest
```

Covered: whitespace-preserving pre-tokenisation, determinism, vocabulary growth, lossless Unicode round-trips, early stopping on `--min-frequency`, save/load, and the compression report.

MIT licensed.
