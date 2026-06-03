"""Per-conversation homogeneity scoring for the WildChat partition experiment.

Homogeneity = how similar a conversation's *user* turns are to each other. A user doing
15 near-identical translations scores high; a topic-switching chat scores low. We score
it with TF-IDF cosine between user turns (sklearn; no extra deps, CPU-only, and
independent of the model whose entropy we are explaining — no circularity).

Used to split WildChat by its *own* homogeneity, holding format and dataset constant, so
heterogeneity can be isolated as the driver of any entropy collapse (vs the
format-confounded DDXPlus contrast).
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def user_messages(msgs: list[dict]) -> list[str]:
    """The user-turn contents of a normalized conversation."""
    return [m["content"] for m in msgs if m["role"] == "user"]


def _tfidf(texts: list[str]):
    """TF-IDF matrix; retry without the English stop-list if it empties the vocab."""
    for stop in ("english", None):
        try:
            return TfidfVectorizer(stop_words=stop).fit_transform(texts)
        except ValueError:
            continue
    return None


def homogeneity_score(user_texts: list[str]) -> float:
    """Mean pairwise TF-IDF cosine across user turns (1 = identical, 0 = disjoint).

    Returns NaN when there are fewer than two non-empty turns or no usable vocabulary.
    """
    texts = [t for t in user_texts if t and t.strip()]
    if len(texts) < 2:
        return float("nan")
    x = _tfidf(texts)
    if x is None:
        return float("nan")
    sim = cosine_similarity(x)
    iu = np.triu_indices(sim.shape[0], k=1)
    return float(np.mean(sim[iu]))


def consecutive_homogeneity(user_texts: list[str]) -> float:
    """Mean TF-IDF cosine between *consecutive* user turns (local topic continuity)."""
    texts = [t for t in user_texts if t and t.strip()]
    if len(texts) < 2:
        return float("nan")
    x = _tfidf(texts)
    if x is None:
        return float("nan")
    sim = cosine_similarity(x)
    return float(np.mean([sim[i, i + 1] for i in range(sim.shape[0] - 1)]))
