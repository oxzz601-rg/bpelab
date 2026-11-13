"""bpelab: byte-pair encoding you can read, train and inspect."""

from .bpe import BASE_VOCAB, Tokenizer, compression_report, pretokenize, train

__version__ = "0.1.0"

__all__ = ["BASE_VOCAB", "Tokenizer", "compression_report", "pretokenize", "train"]
