# BPE tokenizer: training, encoding, decoding and experiments

Reference notes for CS336 Assignment 1, §2 (handout §2.4–2.7). They cover the whole pipeline, from raw text to the `uint16` token arrays the language model will train on. For each step they record what the code does, why it's built that way, and what the measurements showed.

The merge-loop optimisation (full scan → lazy max-heap) is summarised in §3. The full step-by-step heap trace is in [train_bpe_heap.md](train_bpe_heap.md).

---

## 0. Map of the code

| File | What lives there |
|---|---|
| [cs336_basics/pretokenization_example.py](../cs336_basics/pretokenization_example.py) | `PAT` regex, `find_chunk_boundaries`, `pre_tokenizer`, `run_pre_tokenizer` (parallel pre-token counting) |
| [cs336_basics/tokenizer.py](../cs336_basics/tokenizer.py) | `train_bpe` (lazy-heap merge loop), `train_bpe_tinystories`, `train_bpe_expts_owt` |
| [cs336_basics/bpe_tokenizer.py](../cs336_basics/bpe_tokenizer.py) | `Tokenizer`: `from_files`, `encode`, `encode_iterable`, `decode` |
| [tests/adapters.py](../tests/adapters.py) | `run_train_bpe` and `get_tokenizer` wire the code above into the test suite |
| [notebooks/tokenizer_experiments.ipynb](../notebooks/tokenizer_experiments.ipynb) | §2.7 experiments (a)–(d), plus `FastTokenizer` |
| `data/bpe_tinystories/`, `data/bpe_owt/` | Trained `vocab.pkl` (`dict[int, bytes]`) and `merges.pkl` (`list[tuple[bytes, bytes]]`) |
| `data/tokens/*.bin` | Encoded datasets, as raw `uint16` |

The rough prototypes all of this came from are in `study_productivity_tool/notes/pretraining-inference/bpe-tokenizer/rough.ipynb`, under the sections "Encoder" and "Decoder".

---

## 1. The pipeline at a glance

```
                 TRAINING (once per corpus)                          USING THE TOKENIZER
 ┌──────────────────────────────────────────────────┐   ┌───────────────────────────────────────────┐
 raw text ──► split on <|endoftext|> ──► regex pre-tokens ──► count   vocab.pkl + merges.pkl
                                              │                         │
                                              ▼                         ▼
                              {(b'h',b'e',b'l',b'l',b'o'): 1234, ...}   text ──► split specials ──► pre-tokens
                                              │                                      │
                                              ▼                                      ▼
                               merge loop (highest-count pair)         apply merges in creation order
                                              │                                      │
                                              ▼                                      ▼
                                   vocab (id → bytes), merges                 ids ──► uint16 .bin
 └──────────────────────────────────────────────────┘   └───────────────────────────────────────────┘
```

Training and encoding share the same front end: special-token splitting followed by the GPT-2 regex. **Encoding has to pre-tokenize exactly the way training did.** Otherwise the merges get applied to pieces they were never learned on.

---

## 2. Pre-tokenization

### 2.1 Why pre-tokenize at all
Running BPE over the raw byte stream would let merges cross word boundaries (`"dog."` and `"dog!"` would learn different tokens). It would also be very slow, since every merge would rescan the whole corpus. Splitting into **pre-tokens** first gives two benefits:
- merges never cross a pre-token boundary;
- training works on **counts of unique pre-tokens** instead of the raw text. TinyStories has billions of characters but only a few tens of thousands of distinct pre-tokens.

### 2.2 The regex (GPT-2)

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

| Alternative | Matches | Example |
|---|---|---|
| `'(?:[sdmt]\|ll\|ve\|re)` | English contractions | `'m`, `'ll` |
| ` ?\p{L}+` | a word, with at most one leading space | ` years` |
| ` ?\p{N}+` | a number, with at most one leading space | ` 42` |
| ` ?[^\s\p{L}\p{N}]+` | a punctuation/symbol run | `,` `!` |
| `\s+(?!\S)` | whitespace not followed by a non-space; leaves the last space for the next word | `'  '` before ` old` |
| `\s+` | any remaining whitespace | `\n` |

```
"Hello, I'm 42 years   old!" → ['Hello', ',', ' I', "'m", ' 42', ' years', '  ', ' old', '!']
```

The leading space attaches to the word (` years`, not `years`). That's why most learned tokens start with a space (`b' the'`, `b' accomplishment'`).

The `regex` package is needed, not the built-in `re`, because the pattern uses Unicode classes like `\p{L}`.

### 2.3 Special tokens are hard walls
`<|endoftext|>` separates documents. Before the regex runs, the text is split on special tokens, so:
- no pre-token (and hence no merge) ever spans two documents;
- during training, the special token itself is **not counted** as text. It is added to the vocab directly (id 256).

### 2.4 Parallel counting (`run_pre_tokenizer`)
1. `find_chunk_boundaries` picks roughly equal byte offsets, then moves each one forward to the next `<|endoftext|>`. This makes every chunk a set of whole documents.
2. Each worker in a `fork` pool gets only `(start, end)`. It reads and decodes its own span, and returns a `str → count` dict.
3. The parent merges the dicts as they arrive (`imap_unordered`, since order doesn't matter for counts). It then converts each string to a tuple of single-byte `bytes`: `"hi" → (b'h', b'i')`.

For OWT (12 GB on a 16 GB machine), `num_chunks` is set from `chunk_mb=64` rather than the CPU count. That keeps each worker's text and dict small.

---

## 3. BPE training (`train_bpe`)

### 3.1 The rule
Start with `vocab = {0..255: single bytes} + {256: b'<|endoftext|>'}`. Then repeat until `len(vocab) == vocab_size`:

> Merge the adjacent pair with the **highest total count**, breaking ties by the **lexicographically greatest** pair. Add the merged bytes as a new token, and record `(a, b)` in `merges`.

`len(merges) = vocab_size − 257`: 9,743 for TinyStories (10K) and 31,743 for OWT (32K).

### 3.2 Incremental pair counts
Two structures are built once:
- `pair_counts[(a, b)]`: the total frequency of the pair across all words, weighted by each word's count;
- `pair_words[(a, b)]`: the indices of the words containing the pair.

After choosing `best`, only the words in `pair_words[best]` are touched. For each one:
1. subtract all of its old pair counts;
2. rewrite the word with `best` merged;
3. add its new pair counts.

No other word changes, so no other count changes.

### 3.3 Finding `best` with a lazy max-heap
The original code used `max(pair_counts, key=lambda p: (pair_counts[p], p))`, which is O(P) per merge. P (the number of live pairs) grows as merges create new tokens. On TinyStories valid, the run checked **99.7M pairs** in total, almost all of them unchanged.

The replacement:

**(a) Reverse the min-heap.** Entries are `(-count, _Desc(pair))`. `_Desc.__lt__` returns `self.pair > other.pair`, so on equal counts the *greatest* pair pops first. That's a wrapper because `bytes` can't be negated.

**(b) Lazy deletion.** `heapq` can't update an entry in place. So whenever a pair's count changes, a *new* entry is pushed and the old one is left behind. On pop, an entry is checked against the source of truth:

```python
while True:
    neg_cnt, d = heapq.heappop(heap)
    if pair_counts.get(d.pair) == -neg_cnt:   # current? use it
        break                                  # otherwise it's stale: discard
```

**(c) Why it gives the same answer.** The invariant is that *every live pair has at least one heap entry carrying its current count*. All pairs are pushed at the start, and every changed pair is re-pushed after each merge. Stale entries never win. The first valid entry popped is therefore the smallest valid tuple, which is exactly what `max(...)` would pick.

**(d) Bounded growth.** When `len(heap) > 2·len(pair_counts) + 1_000_000`, the heap is rebuilt from `pair_counts`.

Worked example from the handout corpus (`low×5 lower×2 widest×3 newest×6`):

```
merge 1: pop 9 (s,t)                                  (s,t) and (e,s) tie at 9; b's' > b'e'
merge 2: pop 9 (e,st)                                 the stale (e,s) is still in the heap, but b'st' > b's'
merge 3: pop 9 (e,s) stale → 8 (w,e) stale → 8 (w,e) stale dup → 7 (o,w) valid
result : (s,t) (e,st) (o,w) (l,ow) (w,est) (n,e)
```

Measured on TinyStories valid, with the same input to both versions:

| | old `max()` | lazy heap |
|---|---|---|
| work | 99,657,168 pairs checked | 136,201 pushes + 106,542 pops (96,799 stale) |
| merge-loop time | 5.66 s | 0.39 s (~14×) |
| output | 9,743 merges | identical |

The heap's work scales with *how many counts change*, not with *how many pairs exist*.

### 3.4 What got learned
| | TinyStories (10K) | OWT (32K) |
|---|---|---|
| first merges | `' t'`, `'he'`, `' a'`, `' s'`, `' w'`, `'nd'`, `' the'` | `' t'`, `' a'`, `'he'`, `'in'`, `'re'`, `' the'` |
| longest token | `b' accomplishment'` (15 bytes) | 64 bytes of `ÃÂÃÂ…` (UTF-8 mojibake repeated in scraped web text) |

TinyStories's longest token is a real word that children's stories use. OWT's is junk text that happens to be very frequent. Web data is noisy, and BPE faithfully learns that noise.

---

## 4. The `Tokenizer` class (§2.6)

[cs336_basics/bpe_tokenizer.py](../cs336_basics/bpe_tokenizer.py). The merge loop and decode logic are the rough-notebook versions, unchanged. The class adds the interface the spec requires.

### 4.1 Construction
```python
Tokenizer(vocab, merges, special_tokens=None)
Tokenizer.from_files("data/bpe_owt/vocab.pkl", "data/bpe_owt/merges.pkl", ["<|endoftext|>"])
```
- Special tokens missing from the vocab are **appended** with new ids.
- `tok2id` is the reverse map `bytes → id`.
- The special-token split regex is a **capture group**, with tokens sorted **longest first**:
  ```python
  "(" + "|".join(map(re.escape, sorted(specials, key=len, reverse=True))) + ")"
  ```
  - *Capture group*: `re.split` keeps the separators, so the output alternates text and special token: even indices are text, odd indices are specials.
  - *Longest first*: with specials `["<|endoftext|>", "<|endoftext|><|endoftext|>"]`, the double token must match before its prefix does. Regex alternation is first-match, not longest-match. This case is covered by `test_overlapping_special_tokens`.

### 4.2 `encode(text)`
1. Split on specials. Each special becomes `tok2id[special.encode()]` directly, and never goes through the regex or the merges.
2. Run each text piece through `pre_tokenizer_regex` (the same `PAT` as training).
3. Turn each pre-token into single bytes: `[bytes([b]) for b in s.encode("utf-8")]`.
4. `_apply_merges` walks `merges` **in creation order**. For each merge, it scans the list and replaces every adjacent `(m[0], m[1])` with `m[0] + m[1]`.
5. Map each resulting bytes to its id.

The handout's example checks out: with the toy vocab and merges, `encode("the cat ate") == [9, 7, 1, 5, 10, 3]`.

```
'the'  → [t,h,e] →(t,h)→ [th,e] →(th,e)→ [the]        → [9]
' cat' → [ ,c,a,t] →( ,c)→ [ c,a,t]                    → [7, 1, 5]
' ate' → [ ,a,t,e] →( ,a)→ [ a,t,e] →( a,t)→ [ at,e]   → [10, 3]
```

Why "in creation order" matters: merges must be replayed in the order training learned them. A later merge may depend on a token that only exists because of an earlier one: `(th, e)` needs `th`.

### 4.3 `decode(ids)`
```python
b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
```
- Concatenate **all** bytes first, then decode once. A single token can be half of a multi-byte character (e.g. one byte of `é`), so decoding token by token would fail on valid input.
- `errors="replace"`: arbitrary id sequences need not form valid UTF-8. Bad bytes become U+FFFD (`�`) instead of raising.

### 4.4 `encode_iterable(iterable)`
This is a generator for files too big for memory: `for id in tok.encode_iterable(open(path))`. The difficulty is that **encoding chunk by chunk must give exactly the same ids as encoding the whole text.**

The naive approach (`yield from encode(line)` for each line) is wrong. The regex result depends on what comes next:

```
"a \n" + "b"   encoded together  → ['a', ' ', '\n', 'b']
"a \n"         encoded alone     → ['a', ' \n']            ← different pre-tokens, different ids
```

`\s+(?!\S)` swallows `" \n"` at end-of-string, because nothing non-space follows. Once `b` arrives, it has to split differently.

The fix keeps a buffer and holds back the last pre-token:
1. `buf += chunk`, then split on specials. Everything before the final text segment is safe to emit, since specials are hard walls.
2. Run the regex on the final segment. Emit **all matches except the last**, and keep the last one (`tail[matches[-1].start():]`) as the new buffer.
3. After the loop, flush whatever is left with `encode(buf)`.

**Subtle point:** step 2 emits the *match strings themselves* (`m.group()`). It does not re-run the regex on `tail[:cut]`. Re-running it on the shorter prefix hits the same lookahead problem: `"a \n"` would come back as `['a', ' \n']` again.

Memory is bounded by one line plus one pre-token, regardless of file size. **Assumption:** a special token never straddles two chunks. That always holds for file lines, because specials contain no `\n`.

### 4.5 Tests
```bash
uv run pytest tests/test_tokenizer.py      # 23 passed, 2 skipped (rlimit memory tests are Linux-only)
```
The tests compare against `tiktoken`'s GPT-2 encoding, using GPT-2's vocab and merges converted to raw bytes.

---

## 5. `FastTokenizer`: the experimental speed fix (notebook only)

### 5.1 The problem
`_apply_merges` runs `for m in merges` over **every** merge for **every** pre-token: 32K list scans to encode `" the"`, of which maybe 3 merges apply. Measured single-process throughput:

| | TinyStories (10K merges) | OWT (32K merges) |
|---|---|---|
| original | 0.006 MB/s | 0.002 MB/s |

At 2 KB/s, OWT train (12 GB) would take about **70 days**.

### 5.2 The fix
`FastTokenizer(Tokenizer)` overrides two methods and leaves the module untouched.

**(1) Rank-based merging.** It builds `ranks = {pair: index in merges}`, then repeats:
```python
m = min(zip(wrds, wrds[1:]), key=lambda p: ranks.get(p, inf))   # lowest-rank adjacent pair
if m not in ranks: break                                         # nothing left to merge
# merge every occurrence of m (same scan-and-replace loop as the original)
```
The cost is O(len(word)²) per pre-token instead of O(#merges × len(word)), and words are short.

*Why this is identical to "apply all merges in order":* a token created by merge *k* can only appear in pairs whose merges come after *k*, because training could only learn such a pair after the token existed. So at any moment, the lowest-rank pair present is exactly the next merge the original loop would apply. Merges that don't apply are simply skipped instead of scanned.

**(2) Per-pre-token cache.** `cache[" the"] = [id]`. Natural text repeats pre-tokens constantly, so after warm-up almost every lookup is a dict hit. The cache is cleared at 500K entries to bound memory in long runs.

**Verified:** the notebook asserts `FastTokenizer.encode(d) == Tokenizer.encode(d)` on all 10 TinyStories sample docs and on 2 OWT docs, before any fast numbers are used.

### 5.3 Result
| | TinyStories | OWT |
|---|---|---|
| fast, 1 process | 13.6 MB/s | 7.8 MB/s |
| speed-up | ~2,300× | ~3,900× |

OWT is slower than TinyStories for two reasons: it has more distinct pre-tokens, so more cache misses, and longer words.

---

## 6. Experiments (§2.7)

All numbers come from [notebooks/tokenizer_experiments.ipynb](../notebooks/tokenizer_experiments.ipynb):
- the 10-doc samples use `random.seed(0)`;
- TinyStories docs come from the valid split;
- OWT docs come from the first 20 MB of `owt_valid.txt.gz`.

### (a) Compression ratio
Compression ratio = total UTF-8 bytes ÷ total tokens. Higher means each token carries more text.

| Tokenizer → data | bytes/token (10 docs) | bytes/token (full valid split) |
|---|---|---|
| TinyStories 10K → TinyStories | **4.10** | 4.12 |
| OWT 32K → OWT | **4.41** | 4.37 |

The ratios are similar even though OWT's text is far more diverse. The 3.2× larger vocabulary pays for that diversity.

### (b) Mismatched tokenizer
| | bytes/token |
|---|---|
| TinyStories tokenizer → OWT | **3.32** (1.33× more tokens than the OWT tokenizer) |
| OWT tokenizer → TinyStories | 3.96 (only slightly worse than 4.10) |

Qualitatively, words outside the children's-story vocabulary fall apart:

```
TinyStories tok: ' ph','y','s','ic','al'   ' imp','er','f','ect','ions'   ' exc','use'
OWT tok:         ' physical'               ' imperfect','ions'             ' excuse'
```

19.6% of the tokens are single bytes under the TinyStories tokenizer, compared with 14.1% under the OWT one.

The asymmetry is the main lesson. A general tokenizer (OWT) handles simple text almost as well as a specialised one. A specialised tokenizer handles general text badly, so the model sees longer sequences in which each token carries less meaning.

### (c) Throughput and the Pile (825 GB)
| Setup | Throughput | Pile time |
|---|---|---|
| original loop, 1 process | 2–6 KB/s | ~4.5–13.5 **years** |
| `FastTokenizer`, 1 process (OWT) | 7.8 MB/s | ~30 h |
| `FastTokenizer`, 8 processes, end-to-end (OWT valid, includes I/O) | 42 MB/s | **~5.4 h** |

Eight processes gave 5.4× speed-up rather than 8×, because of pool overhead, I/O, and a cold cache in each worker.

### (d) Encoding datasets to `uint16`

**Why this step exists.** Training will draw random windows `x = ids[i:i+L]`, `y = ids[i+1:i+L+1]` millions of times. Tokenizing on the fly would pay the tokenizer cost on every step. Instead, the corpus is tokenized **once** into a flat id array, and training just slices into it.

**Why `uint16`.** `uint16` holds 0–65,535, and the largest ids are 9,999 (TinyStories) and 31,999 (OWT), so nothing is lost. It's 2 bytes per token:

| Storage for OWT train (~2.7B tokens) | Size |
|---|---|
| Python `list[int]` | ≥ 22 GB (8-byte pointers alone): doesn't fit in 16 GB of RAM |
| `int64` / `int32` | ~22 GB / ~11 GB |
| **`uint16`** | **~5.4 GB** |

A vocab larger than 65,536 would need `uint32`. `np.array(ids, dtype=np.uint16)` raises `OverflowError` rather than silently wrapping.

**How `encode_file_to_bin` works**, with each piece tied to a constraint:

| Piece | Constraint it serves |
|---|---|
| `find_chunk_boundaries` at `<\|endoftext\|>` | **Correctness.** Specials are hard walls (§2.3), so chunks encoded separately and concatenated give exactly the whole-file result. Cuts land on an ASCII `<`, so no UTF-8 character is split. |
| Workers get `(start, end)` and read their own span | **Memory.** The parent never loads the 12 GB text. |
| Tokenizer in a global, `fork` pool | **Speed.** Workers inherit it without pickling the vocab, merges and cache for each task. |
| `chunk_mb=32` | **Memory.** Each worker holds a few hundred MB at most. |
| `pool.imap` (ordered, **not** `imap_unordered`) | **Correctness.** The token stream must be in file order. |
| `arr.tofile(out)` per chunk | **Memory.** Ids are streamed to disk and never all held at once. |
| `check_bin` | **Verification.** `max id < vocab size`, and decoding the first 2,000 ids reproduces the start of the file. |

**Output:** raw headerless `uint16`. Load it with:
```python
ids = np.memmap("data/tokens/owt_train.bin", dtype=np.uint16, mode="r")
```
`memmap` doesn't read the file up front. The OS pages in only the slices you touch, so random batches from a 5 GB file cost a few KB of reads each.

| File | Tokens | Size | bytes/token | Time (8 processes) |
|---|---|---|---|---|
| `ts_valid.bin` | 5,465,883 | 10.9 MB | 4.12 | 0.4 s |
| `owt_valid.bin` | 66,401,098 | 132.8 MB | 4.37 | 6.9 s |
| `ts_train.bin` | *not yet run* (`RUN_TRAIN = False`) | ~1 GB expected | | |
| `owt_train.bin` | *not yet run* | ~5.4 GB expected | | |

### Deliverable answers (1–2 sentences each)
- **(a)** The TinyStories tokenizer (10K) achieves 4.10 bytes/token on TinyStories, and the OWT tokenizer (32K) achieves 4.41 bytes/token on OWT.
- **(b)** The TinyStories tokenizer gets only 3.32 bytes/token on OWT (1.33× more tokens), because words outside its children's-story vocabulary break into short fragments (`' physical'` → `' ph','y','s','ic','al'`).
- **(c)** The optimised tokenizer runs at ~7.8 MB/s on one process (42 MB/s with 8), so the 825 GB Pile would take ~30 h single-process or ~5.4 h on 8 cores. The unoptimised merge loop would take years.
- **(d)** Both vocabularies have fewer than 65,536 ids, so every id fits in `uint16` without loss, at half the storage of `int32` and a quarter of `int64`.

---

## 7. Gotchas worth remembering

1. **Pre-tokenize identically in training and encoding.** Use the same `PAT`, and the same split on special tokens before the regex.
2. **Merge order is the algorithm.** Apply merges in creation order, or by lowest rank, which is equivalent. Never apply them by frequency or in arbitrary order.
3. **Decode all bytes, then decode once**, with `errors="replace"`. Tokens aren't character-aligned.
4. **Regex alternation is first-match.** Sort overlapping special tokens longest-first.
5. **`\s+(?!\S)` makes the regex result depend on later text.** Streaming encoders have to hold back the last pre-token and emit the *match strings*, not a re-run of the regex on a prefix.
6. **Ordered `imap` for token streams, `imap_unordered` for counts.** Order matters only when the output is a sequence.
7. **Use `fork` on macOS** (`mp.get_context("fork")`). The default there is `spawn`, which can't see functions defined in a notebook.
8. **Import quirk:** `cs336_basics/tokenizer.py` uses a script-style `from pretokenization_example import ...`. That works when the file is run as a script (`python cs336_basics/tokenizer.py` puts `cs336_basics/` on `sys.path`), but fails when it's imported as a package. `tests/adapters.py` therefore imports `train_bpe` lazily inside `run_train_bpe`, and `test_train_bpe.py` will fail until that import becomes `from cs336_basics.pretokenization_example import ...`.
9. **Profile before scaling.** The original encoder was correct but ~3,000× too slow for (d). Knowing which loop dominates (merges × pre-tokens) made the fix obvious.

---

## 8. How to re-run

```bash
# train tokenizers from the repo root (writes data/bpe_*/vocab.pkl, merges.pkl);
# running the file as a script puts cs336_basics/ on sys.path, which its import needs
uv run python cs336_basics/tokenizer.py        # TinyStories 10K
uv run python cs336_basics/tokenizer.py owt    # OWT 32K (decompresses owt_train.txt.gz first)

# tokenizer tests
uv run pytest tests/test_tokenizer.py

# experiments notebook (headless); or open it in VS Code with the repo's .venv kernel
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/tokenizer_experiments.ipynb
```

To encode the full training sets, set `RUN_TRAIN = True` in the last code cell of the notebook. Expect about 5–10 min and about 6.4 GB of output in `data/tokens/`.
