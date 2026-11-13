"""The ``bpelab`` command line: train, encode, inspect and compare tokenizers."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from . import __version__
from .bpe import Tokenizer, compression_report, train


def _read(path):
    if path == "-":
        return sys.stdin.read()
    return pathlib.Path(path).read_text(encoding="utf-8")


def _show_merges(tokenizer, limit):
    if not tokenizer.merges:
        print("  (no merges learned)")
        return
    for row in tokenizer.merge_table(limit=limit):
        print(
            "  #{rank:<3} {left!r:>12} + {right!r:<12} -> {merged!r}".format(
                rank=row["rank"], left=row["left"], right=row["right"], merged=row["merged"]
            )
        )


def cmd_train(args):
    text = _read(args.corpus)
    try:
        tokenizer = train(text, vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    except ValueError as exc:
        print("bpelab: {}".format(exc), file=sys.stderr)
        return 2
    if args.output:
        tokenizer.save(args.output)
        print("wrote {} ({} tokens)".format(args.output, len(tokenizer)), file=sys.stderr)
    print(json.dumps(compression_report(tokenizer, text), indent=2))
    print("\nfirst merges learned (most frequent first):")
    _show_merges(tokenizer, args.show_merges)
    return 0


def cmd_encode(args):
    tokenizer = Tokenizer.load(args.tokenizer)
    text = args.text if args.text is not None else _read(args.file or "-")
    ids = tokenizer.encode(text)
    if args.ids:
        print(json.dumps(ids))
    elif args.pieces:
        for token_id, piece in zip(ids, tokenizer.segmentation(text)):
            print("{:>6}  {!r}".format(token_id, piece.decode("utf-8", "replace")))
    else:
        print(tokenizer.display(text))
        print("{} token(s) for {} character(s)".format(len(ids), len(text)), file=sys.stderr)
    return 0


def cmd_stats(args):
    tokenizer = Tokenizer.load(args.tokenizer)
    text = _read(args.corpus)
    print(json.dumps(compression_report(tokenizer, text), indent=2))
    return 0


def cmd_demo(args):
    text = _read(args.corpus)
    sizes = [int(value) for value in args.sizes.split(",")]
    print("{:>6} {:>10} {:>12} {:>14}".format("vocab", "tokens", "chars/token", "tokens/byte"))
    for size in sizes:
        tokenizer = train(text, vocab_size=size, min_frequency=args.min_frequency)
        report = compression_report(tokenizer, text)
        print(
            "{:>6} {:>10} {:>12} {:>14}".format(
                len(tokenizer), report["tokens"], report["chars_per_token"], report["tokens_per_byte"]
            )
        )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="bpelab", description="Train and inspect byte-pair encoding tokenizers.")
    parser.add_argument("--version", action="version", version="bpelab {}".format(__version__))
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="learn a tokenizer from a text file")
    train_parser.add_argument("corpus", help="text file, or - for stdin")
    train_parser.add_argument("--vocab-size", type=int, default=512)
    train_parser.add_argument("--min-frequency", type=int, default=2)
    train_parser.add_argument("--show-merges", type=int, default=12, help="how many learned merges to print")
    train_parser.add_argument("-o", "--output", help="save the tokenizer as JSON")
    train_parser.set_defaults(func=cmd_train)

    encode = sub.add_parser("encode", help="tokenize text with a saved tokenizer")
    encode.add_argument("tokenizer")
    encode.add_argument("text", nargs="?", help="text to tokenize (default: stdin)")
    encode.add_argument("--file", help="read the text from a file instead")
    encode.add_argument("--ids", action="store_true", help="print token ids as JSON")
    encode.add_argument("--pieces", action="store_true", help="print each id with its bytes")
    encode.set_defaults(func=cmd_encode)

    stats = sub.add_parser("stats", help="compression report for a corpus")
    stats.add_argument("tokenizer")
    stats.add_argument("corpus")
    stats.set_defaults(func=cmd_stats)

    demo = sub.add_parser("demo", help="compression curve across several vocabulary sizes")
    demo.add_argument("corpus")
    demo.add_argument("--sizes", default="300,400,600,1000")
    demo.add_argument("--min-frequency", type=int, default=2)
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
