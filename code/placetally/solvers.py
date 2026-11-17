"""Other ways to link fragments, used for the comparison rows of the places table.

Each takes a symmetric matrix W of pairwise fragment scores (positive: same place) and returns a label per
fragment. The paper's rule is `places.link_fragments`.
"""

import numpy as np


def greedy(W):
    """The best-link rule on a score matrix: continue if W[t, t-1] > 0, else join the best earlier fragment."""
    n = len(W)
    labels = [0]
    for t in range(1, n):
        if W[t, t - 1] > 0:
            labels.append(labels[t - 1])
            continue
        j = int(np.argmax(W[t, :t]))
        labels.append(labels[j] if W[t, j] > 0 else max(labels) + 1)
    return np.array(labels)


def greedy_additive(W):
    """Greedy additive edge contraction: merge the pair of clusters with the largest positive total score."""
    n = len(W)
    S = W.copy()
    np.fill_diagonal(S, -np.inf)
    active = np.ones(n, bool)
    labels = np.arange(n)
    while True:
        M = np.where(active[:, None] & active[None, :], S, -np.inf)
        np.fill_diagonal(M, -np.inf)
        M = np.triu(M, 1) + np.tril(np.full((n, n), -np.inf))
        i, j = divmod(int(np.argmax(M)), n)
        if M[i, j] <= 1e-12:
            break
        S[i, :] += S[j, :]
        S[:, i] += S[:, j]
        S[i, i] = -np.inf
        active[j] = False
        labels[labels == j] = i
    return labels


def move_refine(W, labels, max_rounds=100):
    """Move single fragments to the cluster that most increases the total score, until nothing moves."""
    n = len(W)
    labels = labels.copy()
    for _ in range(max_rounds):
        moved = False
        for f in range(n):
            clusters = np.unique(labels)
            own = labels[f]
            others = np.arange(n) != f
            link = {c: W[f, others & (labels == c)].sum() for c in clusters}
            best, target = -link[own], -1
            for c in clusters:
                if c != own and link[c] - link[own] > best + 1e-12:
                    best, target = link[c] - link[own], c
            if best > 1e-9:
                labels[f] = target if target != -1 else labels.max() + 1
                moved = True
        if not moved:
            break
    return labels


def correlation_clustering(W):
    """A heuristic for the partition objective: greedy contraction, then single-fragment moves."""
    return move_refine(W, greedy_additive(W))


def average_link(W):
    """Merge the pair of clusters with the largest positive mean score."""
    n = len(W)
    S = W.copy()
    size = np.ones(n)
    active = np.ones(n, bool)
    labels = np.arange(n)
    while True:
        M = S / np.outer(size, size)
        M = np.where(active[:, None] & active[None, :], M, -np.inf)
        M = np.triu(M, 1) + np.tril(np.full((n, n), -np.inf))
        np.fill_diagonal(M, -np.inf)
        i, j = divmod(int(np.argmax(M)), n)
        if not M[i, j] > 1e-12:
            break
        S[i, :] += S[j, :]
        S[:, i] += S[:, j]
        size[i] += size[j]
        active[j] = False
        labels[labels == j] = i
    return labels


def single_link(W):
    """Connected components of the positive scores."""
    n = len(W)
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.nonzero(np.triu(W > 0, 1))):
        parent[root(i)] = root(j)
    return np.array([root(i) for i in range(n)])
