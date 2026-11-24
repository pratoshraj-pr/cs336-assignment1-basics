# `train_bpe`: from a full scan to a lazy max-heap

This note explains the change to the merge loop in [`cs336_basics/tokenizer.py`](../cs336_basics/tokenizer.py). The old loop searched every pair on every merge to find the best one. The new loop keeps the candidates in a heap and only touches the pairs that changed.

## 1. What each merge needs

At each merge step, BPE has to pick:

> the adjacent pair with the **highest count**, breaking ties by the **lexicographically greatest** pair.

Both versions keep the same two structures, built once before the loop starts:

- `pair_counts[(a, b)]`: the total frequency of the pair `(a, b)` across all words.
- `pair_words[(a, b)]`: the indices of the words that contain `(a, b)`.

After choosing the `best` pair, both versions update only the words in `pair_words[best]`. For each of those words, they subtract its old pair counts, merge the pair, and add its new pair counts. That part did not change.

**The only difference is how `best` is found.**

## 2. The old way: scan everything

```python
best = max(pair_counts, key=lambda p: (pair_counts[p], p))
```

This line checks **every distinct pair** on **every merge**. Its cost is O(P) per merge, so a whole run costs O(merges × P).

The problem is that P (the number of distinct pairs) does not stay small. Each merge creates a new token, and that token forms new pairs with its neighbours. Measured on the TinyStories validation set (10k vocab, 9,743 merges):

| | value |
|---|---|
| distinct pairs before the first merge | 932 |
| average pairs checked by `max()` per merge | 10,228 |
| total pairs checked over the run | **99,657,168** |

Almost all of that work is wasted. Each merge changes the counts of only a handful of pairs, yet `max()` re-checks thousands of pairs whose counts did not move.

On OpenWebText (32k merges), P should be far larger, because the text is much more varied. The full-scan cost grows as merges × P, so it becomes a multi-hour loop. This is an estimate: P has not been measured on OWT.

## 3. The new way: a max-heap that only hears about changes

A heap gives the best element in O(log n) without looking at the others. Python's `heapq` has two quirks to work around:

1. **It is a min-heap.** It pops the *smallest* item, but we want the largest count.
2. **It cannot update an entry in place.** When a pair's count changes, there is no cheap way to find its old entry and fix it.

### 3.1 Making a min-heap pop "highest count, then greatest pair"

Each entry is a tuple `(-count, _Desc(pair))`:

```python
class _Desc:
    __slots__ = ("pair",)
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):
        return self.pair > other.pair   # reversed on purpose
```

Python compares tuples one element at a time:

- **First element, `-count`:** a higher count gives a more negative number, which is smaller, so it pops first.
- **Second element, `_Desc(pair)`:** this is only reached when the counts tie. `_Desc` reverses `<`, so the *greatest* pair counts as the "smallest" and pops first.

So the smallest tuple in the heap is exactly the pair the old `max(...)` would have chosen.

Why not negate the pair, as we did with the count? Bytes cannot be negated. A wrapper that flips the comparison is the simplest way to get descending order for any comparable value.

### 3.2 Lazy deletion: push new entries, skip old ones

Instead of updating entries, the loop works like this:

- **When a pair's count changes, push a new entry** with the new count. The old entry stays in the heap, now out of date ("stale").
- **When popping, check the entry against the truth:**

```python
while True:
    neg_cnt, d = heapq.heappop(heap)
    if pair_counts.get(d.pair) == -neg_cnt:   # still accurate?
        break                                  # yes, this is best
    # no: stale (the count moved, or the pair is gone), so drop it and keep popping
```

`pair_counts` is always the source of truth. The heap is just a fast index into it that may hold out-of-date entries.

During a merge, every pair that is touched gets collected in `changed`. After the merge, each one that still has a positive count is pushed once:

```python
for p in changed:
    c = pair_counts.get(p)
    if c:
        heapq.heappush(heap, (-c, _Desc(p)))
```

## 4. Worked example

This is the toy corpus from the assignment handout:

```
low ×5   lower ×2   widest ×3   newest ×6
```

Pair counts before any merge (highest first):

```
9 (s,t)   9 (e,s)   8 (w,e)   7 (o,w)   7 (l,o)   6 (n,e)   6 (e,w)
3 (w,i)   3 (i,d)   3 (d,e)   2 (e,r)
```

The heap starts with one entry for each of these 11 pairs. The trace below is the **actual log** of `heappop` and `heappush` calls made by `train_bpe` on this input.

### Merge 1

```
pop 9 (s,t)          ← valid: pair_counts[(s,t)] == 9
```

`(s,t)` and `(e,s)` are tied at 9. `_Desc` picks the greater pair, and `b's' > b'e'`, so `(s,t)` wins.

The words that contain it are `widest` and `newest`:

```
widest: w i d e s t  →  w i d e st
newest: n e w e s t  →  n e w e st
```

After the merge, `(s,t)` and `(e,s)` drop to 0 and are deleted from `pair_counts`, and the new pair `(e,st)` has count 9. Every pair inside those two words was touched, so each one that still has a count gets pushed:

```
push 3 (w,i)   push 3 (i,d)   push 9 (e,st)   push 8 (w,e)
push 6 (n,e)   push 6 (e,w)   push 3 (d,e)
```

Nothing was removed from the heap. The old `9 (e,s)` entry is still in it, but `(e,s)` no longer exists in `pair_counts`, so that entry is stale.

### Merge 2

```
pop 9 (e,st)         ← valid
```

Two entries have count 9: the stale `(e,s)` and the fresh `(e,st)`. Since `b'st' > b's'`, `(e,st)` sorts first and is valid. The word updates give `w i d est` and `n e w est`. As a result, `(w,e)` drops from 8 to 2, because the `newest` copy is gone and only the one in `lower` remains. A fresh `2 (w,e)` is pushed, along with `6 (w,est)`, `3 (d,est)` and the rest.

### Merge 3: lazy deletion in action

```
pop 9 (e,s)          ← stale: (e,s) no longer exists     → skip
pop 8 (w,e)          ← stale: real count is 2 now        → skip
pop 8 (w,e)          ← stale duplicate (see below)       → skip
pop 7 (o,w)          ← valid                             → merge
```

The heap's top entries were out of date. The loop discards them one at a time until it reaches an entry that matches `pair_counts`. The old loop would have reached the same answer, `(o,w)`, by scanning all the live pairs.

**Why was `(w,e)` at 8 in the heap twice?** In merge 1, `newest` had its pairs subtracted and then re-added. `(w,e)` was in `changed` even though its net count stayed at 8, so a duplicate `8` entry was pushed. Duplicates like this are harmless. They are either valid (the same count) or get skipped once they go stale.

The six merges produced are identical for both versions:

```
(s,t)  (e,st)  (o,w)  (l,ow)  (w,est)  (n,e)
```

## 5. Why the new loop picks the same pair

The heap keeps one rule at all times:

> **Every pair in `pair_counts` has at least one heap entry carrying its current count.**

This rule holds for two reasons:

- **Initially**, every pair is pushed.
- **After that**, any pair whose count changes is in `changed`, so it gets pushed with its new count.

When the loop pops:

1. **Stale entries** (the count doesn't match, or the pair is gone) are discarded, and they never decide anything.
2. **The first valid entry** is the smallest valid tuple in the heap. By the rule above, every live pair has a valid entry, so this one beats all of them by `(count, then greatest pair)`. That is exactly what `max(pair_counts, key=lambda p: (pair_counts[p], p))` returns.

So both versions produce the same merges, in the same order.

## 6. Keeping the heap from growing forever

Stale entries pile up, because each merge pushes new ones and only pops what it needs. To bound memory, the loop rebuilds the heap when stale entries dominate:

```python
if len(heap) > 2 * len(pair_counts) + 1_000_000:
    heap = _build_heap(pair_counts)   # one fresh entry per live pair
```

A rebuild costs O(P), the same as one old-style `max()`. It only happens after many merges, so its cost spread across those merges is small. The `+ 1_000_000` slack means small runs never rebuild. For example, the TinyStories validation run peaks at about 30k entries.

## 7. Cost comparison

| | old (`max` over all pairs) | new (lazy heap) |
|---|---|---|
| finding `best` | O(P) every merge | O((1 + stale pops) · log H) |
| after a merge | — | O(changed pairs · log H) pushes |
| whole run | O(merges × P) | O((pushes + pops) · log H) |

Here P is the number of live pairs and H is the heap size. The key point is that the new loop's work depends on **how many counts change**, not on **how many pairs exist**.

Measured on the TinyStories validation set, with the same pre-tokenized input given to both versions:

| | old | new |
|---|---|---|
| work units | 99,657,168 pairs checked by `max()` | 136,201 pushes + 106,542 pops (96,799 of them stale) |
| merge-loop time | 5.66 s | 0.39 s (**~14× faster**) |
| merges / vocab | 9,743 / 10,000 | identical |

The speedup should grow with the corpus. On a larger, more varied corpus like OWT, P grows a lot, but the number of pairs each merge changes grows much more slowly.

## 8. What did not change

- **The per-word update:** subtract the old pairs, merge, add the new pairs.
- **`pair_words`:** it still limits each merge to the words that contain `best`.
- **Tie-breaking:** still the highest count, then the lexicographically greatest pair.
- **The output:** identical `vocab` and `merges`. This is checked by `tests/test_train_bpe.py`, which compares merges exactly against the reference, and by a direct comparison with the old version's output on the TinyStories validation set.
