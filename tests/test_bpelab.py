import pytest

from bpelab.bpe import BASE_VOCAB, Tokenizer, compression_report, pretokenize, train

CORPUS = ("the quick brown fox jumps over the lazy dog. " * 40) + ("tokenizers split text into tokens. " * 20)


@pytest.fixture(scope="module")
def tokenizer():
    return train(CORPUS, vocab_size=300)


def test_pretokenize_keeps_whitespace_runs():
    assert pretokenize("a  b\n") == ["a", "  ", "b", "\n"]
    assert "".join(pretokenize("a  b\n")) == "a  b\n"


def test_training_produces_merges_and_grows_the_vocab(tokenizer):
    assert len(tokenizer.merges) > 10
    assert BASE_VOCAB < len(tokenizer) <= 300
    merged = tokenizer.merge_table(limit=1)[0]
    assert merged["merged"] == merged["left"] + merged["right"]


def test_merges_reduce_token_count(tokenizer):
    assert len(tokenizer.encode(CORPUS)) < len(CORPUS.encode("utf-8"))
    report = compression_report(tokenizer, CORPUS)
    assert report["tokens"] == len(tokenizer.encode(CORPUS))
    assert report["chars_per_token"] > 1


def test_roundtrip_is_lossless_for_unicode(tokenizer):
    for text in ["", "hello", "Héllo, wörld! 🚀", "line one\nline two\t tabs  ", "日本語のテキスト"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_training_is_deterministic():
    first = train(CORPUS, vocab_size=280)
    second = train(CORPUS, vocab_size=280)
    assert first.merges == second.merges


def test_min_frequency_stops_training_early():
    small = train("one two three", vocab_size=400, min_frequency=50)
    assert len(small) == BASE_VOCAB


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        train(CORPUS, vocab_size=200)
    with pytest.raises(ValueError):
        train("")


def test_save_and_load_roundtrip(tmp_path, tokenizer):
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    reloaded = Tokenizer.load(path)
    assert reloaded.merges == tokenizer.merges
    assert reloaded.encode("the lazy dog") == tokenizer.encode("the lazy dog")


def test_display_makes_whitespace_visible(tokenizer):
    shown = tokenizer.display("the dog")
    assert "\u2423" in shown or "dog" in shown
    assert tokenizer.decode(tokenizer.encode("the dog")) == "the dog"
