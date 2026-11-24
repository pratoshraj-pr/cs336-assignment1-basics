import pickle
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.pretokenization_example import pre_tokenizer_regex


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        existing = set(self.vocab.values())
        for tok in self.special_tokens:
            tok_b = tok.encode("utf-8")
            if tok_b not in existing:
                self.vocab[len(self.vocab)] = tok_b
                existing.add(tok_b)

        self.tok2id = dict([(tok, id) for id, tok in self.vocab.items()])

        # longest first so overlapping specials (e.g. "<|eot|><|eot|>") win over their prefixes
        self.special_pat = None
        if self.special_tokens:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pat = re.compile("(" + "|".join(map(re.escape, specials)) + ")")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, wrds):
        for m in self.merges:
            new_w = []
            i = 0
            while i < len(wrds):
                if i + 1 < len(wrds) and wrds[i] == m[0] and wrds[i + 1] == m[1]:
                    new_w.append(m[0] + m[1])
                    i += 2
                else:
                    new_w.append(wrds[i])
                    i += 1
            wrds = new_w
        return wrds

    def _encode_pretoken(self, pre_tok):
        wrds = [bytes([w]) for w in pre_tok.encode("utf-8")]
        return [self.tok2id[tok] for tok in self._apply_merges(wrds)]

    def _encode_text(self, text):
        ids = []
        for wrd in pre_tokenizer_regex(text):
            ids.extend(self._encode_pretoken(wrd.group()))
        return ids

    def _split_special(self, text):
        # odd indices are special tokens, even indices are the text between them
        if self.special_pat is None:
            return [text]
        return self.special_pat.split(text)

    def encode(self, text: str) -> list[int]:
        ids = []
        for i, part in enumerate(self._split_special(text)):
            if i % 2:
                ids.append(self.tok2id[part.encode("utf-8")])
            elif part:
                ids.extend(self._encode_text(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # The last pre-token of the buffer is held back since the next chunk may extend it
        # (e.g. "\n" + "\n"). Assumes special tokens don't straddle chunks, which holds for file lines.
        buf = ""
        for chunk in iterable:
            buf += chunk
            parts = self._split_special(buf)
            for i, part in enumerate(parts[:-1]):
                if i % 2:
                    yield self.tok2id[part.encode("utf-8")]
                else:
                    yield from self._encode_text(part)

            tail = parts[-1]
            matches = list(pre_tokenizer_regex(tail))
            # emit matched strings directly: re-running the regex on a prefix can change the
            # last whitespace pre-token's split because of the (?!\S) lookahead
            for m in matches[:-1]:
                yield from self._encode_pretoken(m.group())
            buf = tail[matches[-1].start() :] if matches else tail
        if buf:
            yield from self.encode(buf)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[tok] for tok in ids).decode("utf-8", errors="replace")
