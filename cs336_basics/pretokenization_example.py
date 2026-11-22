import os
from typing import BinaryIO
from collections import defaultdict
from functools import partial
import regex as re
import multiprocessing as mp

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenizer_regex(txt):
    return re.finditer(PAT, txt)


def pre_tokenizer(txt, spl_tok="<|endoftext|>"):
    tok_cnt = {}
    pattern = "|".join(map(re.escape, spl_tok))
    txt = re.split(pattern, txt)

    for doc in txt:
        for match in pre_tokenizer_regex(doc):
            tok_cnt[match.group()] = tok_cnt.get(match.group(), 0) + 1
    return tok_cnt


def pre_tokenize_chunk(span, data_path, spl_tok):
    # each worker reads its own byte range, so the parent never holds the whole file
    start, end = span
    with open(data_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pre_tokenizer(chunk, spl_tok)


def run_pre_tokenizer(data_path, spl_tok, num_processes=4, num_chunks=None):
    num_chunks = num_chunks or num_processes
    with open(data_path, "rb") as f:
        if spl_tok:
            boundaries = find_chunk_boundaries(f, num_chunks, spl_tok[0].encode("utf-8"))
        else:
            boundaries = [0, f.seek(0, os.SEEK_END)]

    spans = list(zip(boundaries[:-1], boundaries[1:]))
    pre_tokenizer_func = partial(pre_tokenize_chunk, data_path=data_path, spl_tok=spl_tok)

    # merge counts as chunks finish instead of collecting every chunk's dict first
    str_counts = defaultdict(int)
    with mp.get_context("fork").Pool(processes=num_processes) as pool:
        for res in pool.imap_unordered(pre_tokenizer_func, spans):
            for tok, cnt in res.items():
                str_counts[tok] += cnt

    data_pre_tok = defaultdict(int)
    for tok, cnt in str_counts.items():
        data_pre_tok[tuple([bytes([t]) for t in tok.encode("utf-8")])] += cnt

    return data_pre_tok
