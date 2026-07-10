"""Task — learn word embeddings on *Flatland* with word2vec (the baseline).

This is a **capstone**, like ``task_binary_classification.py``: a complete,
runnable script that trains something real end to end on project components.
Where that task trained a classifier on tabular data, this one learns **word
embeddings** from raw text — the *unsupervised* half of what a language model
needs, and the direct ancestor of the token-embedding table BERT later trains.

What is word2vec?
-----------------
An embedding is a dense vector per word (a row of an ``nn.Embedding`` table). We
want words that appear in *similar contexts* to end up with *similar vectors*
("you shall know a word by the company it keeps", Firth 1957). word2vec
(Mikolov et al., 2013) turns that idea into a supervised game played on
unlabelled text:

    skip-gram: from a **center** word, predict the **context** words around it.

We use the efficient variant, **skip-gram with negative sampling (SGNS)**.
Instead of a full softmax over the whole vocabulary (expensive), each training
example is a tiny binary problem:

    * a **positive** pair  (center, real neighbour)      -> should score HIGH
    * a few **negative** pairs (center, random word)     -> should score LOW

The "score" is just the dot product of two vectors, squashed by a sigmoid into a
probability. Maximising the log-probability of the positives and of *not* the
negatives is the loss below. Two separate tables are learned — an **input**
(center) table ``v`` and an **output** (context) table ``u`` — which is standard
word2vec; the input table is the one we keep as "the word vectors".

Per pair the loss is (σ is the logistic sigmoid, k the number of negatives):

    L = - log σ(v_center · u_context)  -  Σ_neg log σ(- v_center · u_neg)

Only dot products, a sigmoid, a log, a sum — all differentiable ops the engine
already has, so ONE ``loss.backward()`` fills the gradient of *both* embedding
tables and ``optim.Adam`` nudges them. No new engine code.

How we measure that it worked
-----------------------------
Good embeddings should place *semantically related* words near each other. We
pick a small **representative set** of Flatland words split into meaning groups
(the geometric shapes, the social/gender words, the space/dimension words) and
score how well-separated those groups are with the **silhouette score** — a
number in ``[-1, 1]`` where higher means "tight, well-separated clusters".

We print a per-word silhouette report (each word's own coefficient with a bar on
a shared ``-1..+1`` scale, grouped by cluster) **before** training (random
vectors — no structure, so ≈ 0 or negative) and **after** training (the groups
should pull apart, so most coefficients and the mean rise). That before/after
jump is the whole point of the exercise. The report is plain text: the coloured
``learn.console`` styling is reserved for the ``learn/`` walkthroughs.

Conventions
-----------
Row-oriented, like ``nn.Embedding``: a batch of ids ``(B,)`` embeds to ``(B, D)``.
Pure NumPy in, plain floats out; reproducible under ``cpu.set_seed``.

HOW TO RUN
==========
    python -m exercises.task_learn_embedding
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# Make the script runnable directly *and* as a module (see the note in
# task_binary_classification.py): add the repo root so ``datasets``, ``bert_cpu``
# and ``learn`` all resolve regardless of how Python was invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
from bert_cpu import engine as cpu
from bert_cpu import nn
from bert_cpu import optim
from bert_cpu.tokenizer import Tokenizer


# ============================================================================ #
# Hyper-parameters (kept small so the pure-NumPy engine finishes in seconds)
# ============================================================================ #
EMBED_DIM = 32          # size D of each word vector
WINDOW = 3              # how many words on each side count as "context"
NEG_K = 5               # negative samples drawn per positive pair
MIN_COUNT = 5           # ignore words rarer than this (too few contexts to learn)
EPOCHS = 100
BATCH = 2048
LR = 0.02               # Adam learning rate

# Representative words, grouped by the meaning we *expect* to emerge. Words not
# present in the learned vocabulary are dropped automatically before scoring.
REPRESENTATIVE: Dict[str, List[str]] = {
    "shapes": ["triangle", "square", "pentagon", "hexagon", "circle", "polygon"],
    "people": ["women", "men", "man", "woman", "wife", "husband"],
    "space":  ["dimension", "dimensions", "space", "plane", "solid", "line"],
}


# ============================================================================ #
# Corpus -> integer word ids
# ============================================================================ #
def build_word_vocab(
    sentences: List[str], min_count: int
) -> Tuple[Dict[str, int], List[str], np.ndarray]:
    """Turn raw sentences into a word-level vocabulary and its frequencies.

    word2vec works on whole words (not the WordPiece subwords the Tokenizer
    produces), so we reuse only the Tokenizer's *basic* pre-tokeniser (lowercase +
    split, punctuation separated), keep alphabetic tokens, and drop any word
    seen fewer than ``min_count`` times — rare words simply don't appear in enough
    contexts to learn a meaningful vector.

    Returns ``(word2id, id2word, freqs)`` where ``freqs[i]`` is the corpus count
    of word id ``i`` (used later to build the negative-sampling distribution).
    """
    counts: Dict[str, int] = {}
    for sent in sentences:
        for tok in Tokenizer._basic_tokenize(sent):
            if tok.isalpha():                        # skip punctuation / numbers
                counts[tok] = counts.get(tok, 0) + 1

    # Keep words above the frequency floor, most-frequent first (so common words
    # get the low ids — purely cosmetic, but conventional).
    kept = sorted(
        (w for w, c in counts.items() if c >= min_count),
        key=lambda w: (-counts[w], w),
    )
    word2id = {w: i for i, w in enumerate(kept)}
    id2word = kept
    freqs = np.array([counts[w] for w in kept], dtype=np.float64)
    return word2id, id2word, freqs


def encode_sentences(sentences: List[str], word2id: Dict[str, int]) -> List[np.ndarray]:
    """Map each sentence to an array of in-vocabulary word ids (rare words dropped)."""
    encoded: List[np.ndarray] = []
    for sent in sentences:
        ids = [word2id[t] for t in Tokenizer._basic_tokenize(sent)
               if t.isalpha() and t in word2id]
        if len(ids) >= 2:                            # need at least one pair
            encoded.append(np.array(ids, dtype=np.int64))
    return encoded


def make_skipgram_pairs(
    encoded: List[np.ndarray], window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate all (center, context) pairs within ``window`` words.

    For every position in every sentence, each of the up-to ``window`` words on
    either side is a positive context. This is the supervised signal: "these two
    words really did occur together".
    """
    centers: List[int] = []
    contexts: List[int] = []
    for ids in encoded:
        n = len(ids)
        for i in range(n):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    centers.append(ids[i])
                    contexts.append(ids[j])
    return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)


def make_negative_sampler(freqs: np.ndarray):
    """Build a sampler over the *unigram distribution raised to the 3/4 power*.

    word2vec draws negatives not uniformly and not by raw frequency, but from
    ``freq ** 0.75`` normalised — a sweet spot that samples rare words a bit more
    often than their frequency alone would, which empirically gives better
    vectors (Mikolov et al., 2013). Returns a function ``sample(shape) -> ids``.
    """
    weights = freqs ** 0.75
    probs = weights / weights.sum()

    def sample(shape) -> np.ndarray:
        return np.random.choice(len(probs), size=shape, p=probs)

    return sample


# ============================================================================ #
# The model — two embedding tables
# ============================================================================ #
class SkipGramNS(nn.Module):
    """Skip-gram with negative sampling: an input (center) and output (context) table.

    Both are plain ``nn.Embedding`` tables of shape ``(vocab, D)``. There is no
    ``forward`` in the usual sense — the training objective (below) reaches into
    both tables directly — so we expose them as attributes and let ``Module``
    collect their ``Parameter`` weights for the optimiser.
    """

    def __init__(self, vocab_size: int, dim: int) -> None:
        self.in_emb = nn.Embedding(vocab_size, dim)      # the kept "word vectors"
        self.out_emb = nn.Embedding(vocab_size, dim)     # context/output vectors


def log_sigmoid(x: cpu.Tensor) -> cpu.Tensor:
    """Numerically-gentle ``log σ(x)`` built from engine primitives.

    ``σ(x) = 1 / (1 + e^{-x})``; we take its log. The engine has no ``sigmoid``,
    but ``exp``, ``+``, ``** -1`` and ``log`` are enough to *compose* it — and
    because each of those carries its own backward rule, the gradient flows
    through automatically. Our dot-product scores stay in a modest range on this
    corpus, so the plain form is stable in the engine's float64.
    """
    sigmoid = ((-x).exp() + 1.0) ** -1.0
    return sigmoid.log()


def ns_loss(
    model: SkipGramNS,
    centers: np.ndarray,
    contexts: np.ndarray,
    negatives: np.ndarray,
) -> cpu.Tensor:
    """The skip-gram negative-sampling loss for one batch (a single graph).

    Shapes: ``centers`` / ``contexts`` are ``(B,)``; ``negatives`` is ``(B, K)``.
    We look up vectors, score by dot product, and sum the log-sigmoids::

        pos_score = v · u_context                       # want σ(pos) -> 1
        neg_score = v · u_neg   (per negative)          # want σ(-neg) -> 1
        L = -mean_over_batch[ log σ(pos) + Σ_k log σ(-neg_k) ]

    Returning a scalar ``Tensor`` means one ``.backward()`` fills the gradients of
    both embedding tables at once.
    """
    b = centers.shape[0]

    v = model.in_emb(centers)                    # (B, D)  center vectors
    u_pos = model.out_emb(contexts)              # (B, D)  true-context vectors
    u_neg = model.out_emb(negatives)             # (B, K, D) negative vectors

    # Positive score: elementwise product summed over the D dimension -> (B,).
    pos_score = (v * u_pos).sum(axis=1)

    # Negative scores: broadcast the center vector against the K negatives.
    v3 = v.reshape(b, 1, EMBED_DIM)              # (B, 1, D)
    neg_score = (u_neg * v3).sum(axis=2)         # (B, K)

    # Mean over the batch of (log σ(pos) + Σ log σ(-neg)); negate for a loss.
    total = log_sigmoid(pos_score).sum() + log_sigmoid(-neg_score).sum()
    return -total / b


# ============================================================================ #
# Silhouette score (pure NumPy, cosine distance)
# ============================================================================ #
def silhouette_samples(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-point silhouette coefficient of ``vectors`` under integer ``labels``.

    For each point ``i`` with ``a`` = mean distance to its *own* group and ``b`` =
    smallest mean distance to *another* group, the coefficient is
    ``(b - a) / max(a, b)`` — near ``+1`` when a point sits snugly in its cluster
    and far from the others, near ``0`` on a boundary, negative when misplaced.
    We use **cosine** distance (``1 - cosine similarity``), the natural metric for
    embeddings, where only direction matters.

    Returns the array of per-point coefficients (mean it for the overall score).
    """
    X = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    dist = 1.0 - X @ X.T                          # cosine distance matrix
    np.fill_diagonal(dist, 0.0)

    labels = np.asarray(labels)
    uniq = set(labels.tolist())
    scores = np.zeros(len(labels))
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():                        # a singleton group has no 'a'
            continue
        a = dist[i, same].mean()
        b = min(
            dist[i, labels == lab].mean()
            for lab in uniq if lab != labels[i]
        )
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return scores


def silhouette_score(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Overall silhouette: the mean of the per-point coefficients."""
    return float(silhouette_samples(vectors, labels).mean())


def representative_data(
    model: SkipGramNS, word2id: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[str, List[str]]]]:
    """Gather the embedding rows for the representative words that are in-vocab.

    Returns ``(vectors, labels, words_flat, clusters)`` where ``clusters`` is the
    list of ``(group_name, kept_words)`` actually used (groups with < 2 in-vocab
    words are skipped), ``labels`` is the contiguous cluster index per word, and
    ``words_flat`` is the parallel list of word strings — so the report can print
    each word beside its own silhouette coefficient.
    """
    W = model.in_emb.weight.data                  # (vocab, D) current word vectors
    clusters: List[Tuple[str, List[str]]] = []
    for name, words in REPRESENTATIVE.items():
        kept = [w for w in words if w in word2id]
        if len(kept) >= 2:                        # a group needs >= 2 to matter
            clusters.append((name, kept))

    vecs: List[np.ndarray] = []
    labels: List[int] = []
    words_flat: List[str] = []
    for ci, (_, kept) in enumerate(clusters):
        for w in kept:
            vecs.append(W[word2id[w]])
            labels.append(ci)
            words_flat.append(w)
    return np.array(vecs), np.array(labels), words_flat, clusters


# ============================================================================ #
# Silhouette report (plain text — the console colours are reserved for learn/viz)
# ============================================================================ #
# Layout constants for the per-word report.
_WORD_W = 16            # width of the word column
_VAL_W = 8             # width of the coefficient column
_HALF = 25             # half-width of the [-1, +1] axis, so a full axis is 50 cells
_BAR_UNIT = 1.0 / _HALF  # silhouette value represented by one bar cell (= 0.04)
_TOTAL_W = _WORD_W + _VAL_W + 2 + (2 * _HALF + 1)


def _bar(coeff: float) -> str:
    """Draw one coefficient as a horizontal bar on a fixed ``[-1, +1]`` axis.

    A ``|`` marks the zero baseline (always the same column so bars line up).
    Positive coefficients grow ``#`` cells to the right, negative to the left, one
    cell per ``_BAR_UNIT`` (0.04) of silhouette.
    """
    cells = [" "] * (2 * _HALF + 1)
    cells[_HALF] = "|"
    n = int(round(coeff / _BAR_UNIT))
    n = max(-_HALF, min(_HALF, n))
    if n > 0:
        for k in range(1, n + 1):
            cells[_HALF + k] = "█"
    elif n < 0:
        for k in range(1, -n + 1):
            cells[_HALF - k] = "█"
    return "".join(cells).rstrip() or "|"


def _scale_header() -> str:
    """The ``-1 ... 0 ... +1`` axis header, aligned with the bars below it."""
    axis = [" "] * (2 * _HALF + 1)
    for pos, txt in ((0, "-1"), (_HALF, "0"), (2 * _HALF - 1, "+1")):
        for k, ch in enumerate(txt):
            axis[pos + k] = ch
    prefix = "Escala:".ljust(_WORD_W) + " " * (_VAL_W + 2)
    return prefix + "".join(axis).rstrip()


def print_silhouette_report(
    title: str,
    vectors: np.ndarray,
    labels: np.ndarray,
    words_flat: List[str],
    clusters: List[Tuple[str, List[str]]],
) -> None:
    """Print the per-word silhouette breakdown, grouped by cluster.

    Shows the overall mean, a shared ``-1..+1`` scale, then every word with its
    own coefficient and a bar — the standard way to read a silhouette, where you
    can spot which individual words sit well inside their group and which straddle
    a boundary (small or negative bar).
    """
    coeffs = silhouette_samples(vectors, labels)

    print(title)
    print(f"Silhouette score médio: {coeffs.mean():+.3f}")
    print()
    print(_scale_header())
    print("-" * _TOTAL_W)

    for ci, (name, _) in enumerate(clusters):
        print()
        print(f"Cluster {ci} ({name})")
        print("-" * _TOTAL_W)
        for i in range(len(words_flat)):
            if labels[i] != ci:
                continue
            row = f"{words_flat[i]:<{_WORD_W}}{coeffs[i]:>+{_VAL_W}.3f}  {_bar(coeffs[i])}"
            print(row.rstrip())
    print()


def print_neighbours(model: SkipGramNS, word2id, id2word, words, k: int = 5) -> None:
    """Bonus: nearest neighbours by cosine, to make the learned structure tangible."""
    W = model.in_emb.weight.data
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    print(f"Nearest neighbours after training (cosine, top {k}):")
    for w in words:
        if w not in word2id:
            continue
        sims = Wn @ Wn[word2id[w]]
        order = [j for j in np.argsort(-sims) if j != word2id[w]][:k]
        near = ", ".join(f"{id2word[j]} ({sims[j]:+.2f})" for j in order)
        print(f"   {w:<9} -> {near}")
    print()


# ============================================================================ #
# Training
# ============================================================================ #
def train(
    model: SkipGramNS,
    centers: np.ndarray,
    contexts: np.ndarray,
    sample_negatives,
    epochs: int,
    batch: int,
    lr: float,
) -> None:
    """Mini-batch Adam over the negative-sampling loss; print mean loss per epoch.

    There are hundreds of thousands of pairs, so — unlike the full-batch Adult
    task — we shuffle and step in mini-batches. Each batch draws its own fresh
    negatives, the canonical word2vec recipe.
    """
    opt = optim.Adam(model.parameters(), lr=lr)
    n_pairs = centers.shape[0]

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n_pairs)     # reshuffle every epoch
        running = 0.0
        n_batches = 0
        for start in range(0, n_pairs, batch):
            idx = perm[start:start + batch]
            c = centers[idx]
            ctx = contexts[idx]
            neg = sample_negatives((len(idx), NEG_K))

            opt.zero_grad()
            loss = ns_loss(model, c, ctx, neg)
            loss.backward()
            opt.step()

            running += float(loss.data)
            n_batches += 1

        print(f"   epoch {epoch}/{epochs}   mean loss = {running / n_batches:+.3f}")


# ============================================================================ #
# Entry point
# ============================================================================ #
def main() -> None:
    print("=" * _TOTAL_W)
    print("word2vec on Flatland - learning word embeddings from raw text")
    print("=" * _TOTAL_W)

    cpu.set_seed(0)

    # --- corpus ---------------------------------------------------------- #
    flat = datasets.load_flatland()
    word2id, id2word, freqs = build_word_vocab(flat.sentences, MIN_COUNT)
    encoded = encode_sentences(flat.sentences, word2id)
    centers, contexts = make_skipgram_pairs(encoded, WINDOW)
    sample_negatives = make_negative_sampler(freqs)

    print(
        f"\nCorpus: {len(flat.sentences)} sentences  ->  "
        f"{len(word2id)} words (min_count={MIN_COUNT})  ->  "
        f"{centers.shape[0]:,} skip-gram pairs (window={WINDOW})\n"
    )

    model = SkipGramNS(len(word2id), EMBED_DIM)

    # --- silhouette BEFORE (random init) --------------------------------- #
    vecs, labels, words_flat, clusters = representative_data(model, word2id)
    print_silhouette_report("BEFORE TRAINING (random embeddings)",
                            vecs, labels, words_flat, clusters)

    # --- train ----------------------------------------------------------- #
    print("Training (skip-gram + negative sampling, Adam):")
    train(model, centers, contexts, sample_negatives, EPOCHS, BATCH, LR)
    print()

    # --- silhouette AFTER ------------------------------------------------ #
    vecs, labels, words_flat, clusters = representative_data(model, word2id)
    print_silhouette_report("AFTER TRAINING (learned embeddings)",
                            vecs, labels, words_flat, clusters)

    # --- bonus: nearest neighbours -------------------------------------- #
    print_neighbours(model, word2id, id2word, ["square", "circle", "woman"])


if __name__ == "__main__":
    main()
