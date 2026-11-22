import gzip
import heapq
import os
import pickle
import resource
import shutil
import sys
import time
from collections import defaultdict

from pretokenization_example import run_pre_tokenizer


class _Desc:
    # reverses ordering so the min-heap pops the lexicographically greatest pair on count ties
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair


def _build_heap(pair_counts):
    heap = [(-c, _Desc(p)) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    return heap


def train_bpe(txt_b_c, vocab, vocab_size):
    words = [list(w) for w in txt_b_c]
    freqs = list(txt_b_c.values())

    pair_counts = defaultdict(int)
    pair_words = defaultdict(set)
    for idx, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_counts[p] += freqs[idx]
            pair_words[p].add(idx)

    # lazy max-heap: entries whose count no longer matches pair_counts are skipped on pop
    heap = _build_heap(pair_counts)

    merges = []
    while len(vocab) < vocab_size and pair_counts:
        while True:
            neg_cnt, d = heapq.heappop(heap)
            if pair_counts.get(d.pair) == -neg_cnt:
                break
        best = d.pair
        merges.append(best)
        new_tok = best[0] + best[1]
        vocab[len(vocab)] = new_tok

        changed = set()
        for idx in list(pair_words.pop(best)):
            w, f = words[idx], freqs[idx]

            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                changed.add(p)
                if pair_counts[p] <= 0:
                    del pair_counts[p]

            new_w, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == best[0] and w[i + 1] == best[1]:
                    new_w.append(new_tok)
                    i += 2
                else:
                    new_w.append(w[i])
                    i += 1
            words[idx] = new_w

            for p in zip(new_w, new_w[1:]):
                pair_counts[p] += f
                pair_words[p].add(idx)
                changed.add(p)

        for p in changed:
            c = pair_counts.get(p)
            if c:
                heapq.heappush(heap, (-c, _Desc(p)))

        # drop stale entries once they dominate the heap
        if len(heap) > 2 * len(pair_counts) + 1_000_000:
            heap = _build_heap(pair_counts)

    return vocab, merges, {tuple(w): f for w, f in zip(words, freqs)}


def _train_and_save(input_path, vocab_size, special_tokens, out_dir, num_processes=4, num_chunks=None):
    special_tokens = list(special_tokens)
    start = time.time()

    data_b_c = run_pre_tokenizer(
        input_path, spl_tok=special_tokens, num_processes=num_processes, num_chunks=num_chunks
    )
    pretok_time = time.time() - start

    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    vocab, merges, _ = train_bpe(data_b_c, vocab, vocab_size)
    elapsed = time.time() - start

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(out_dir, "merges.pkl"), "wb") as f:
        pickle.dump(merges, f)

    # ru_maxrss is bytes on macOS, KiB on Linux; pre-tokenization workers count as children
    unit = 1 if sys.platform == "darwin" else 1024
    peak_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit / 2**30
    peak_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * unit / 2**30
    longest = max(vocab.values(), key=len)

    print(f"time: {elapsed:.1f}s ({elapsed / 60:.2f} min); pre-tokenization {pretok_time:.1f}s, merges {elapsed - pretok_time:.1f}s")
    print(f"peak memory: main {peak_self:.2f} GiB, largest worker {peak_child:.2f} GiB")
    print(f"unique pre-tokens: {len(data_b_c)}, vocab size: {len(vocab)}, merges: {len(merges)}")
    print(f"longest token ({len(longest)} bytes): {longest!r}")
    return vocab, merges


def train_bpe_tinystories(
    input_path="data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10_000,
    special_tokens=("<|endoftext|>",),
    out_dir="data/bpe_tinystories",
):
    return _train_and_save(input_path, vocab_size, special_tokens, out_dir)


def _decompress(gz_path):
    # the pre-tokenizer seeks to chunk boundaries, which a .gz stream can't do cheaply
    txt_path = gz_path[: -len(".gz")]
    if not os.path.exists(txt_path):
        print(f"decompressing {gz_path} -> {txt_path}")
        tmp_path = txt_path + ".part"
        with gzip.open(gz_path, "rb") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 2**20)
        os.replace(tmp_path, txt_path)
    return txt_path


def train_bpe_expts_owt(
    input_path="data/owt_train.txt.gz",
    vocab_size=32_000,
    special_tokens=("<|endoftext|>",),
    out_dir="data/bpe_owt",
    num_processes=os.cpu_count(),
    chunk_mb=64,
):
    if input_path.endswith(".gz"):
        input_path = _decompress(input_path)

    # many small chunks keep each worker's text + counts small (~12 GB file, 16 GB RAM)
    num_chunks = max(num_processes, os.path.getsize(input_path) // (chunk_mb * 2**20))
    return _train_and_save(input_path, vocab_size, special_tokens, out_dir, num_processes, num_chunks)


if __name__ == "__main__":
    if sys.argv[1:] == ["owt"]:
        train_bpe_expts_owt()
    else:
        train_bpe_tinystories()
