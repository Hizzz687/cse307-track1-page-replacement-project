"""Classical and learned page-replacement policies for the CSE-307 term paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PolicyResult:
    name: str
    hits: int
    faults: int
    total: int
    split_index: int

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def segment(self, start: int, end: int) -> dict[str, float | int]:
        requests = end - start
        segment_faults = self._faults[start:end]
        faults = sum(segment_faults)
        return {
            "requests": requests,
            "hits": requests - faults,
            "faults": faults,
            "hit_ratio": (requests - faults) / requests if requests else 0.0,
        }

    _faults: tuple[int, ...] = ()


class DecisionTreeClassifier:
    """Small binary decision tree used to avoid hiding the learned component."""

    def __init__(self, max_depth: int = 3, min_samples: int = 8) -> None:
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root: tuple | None = None

    @staticmethod
    def _gini(labels: Sequence[int]) -> float:
        if not labels:
            return 0.0
        positive = sum(labels) / len(labels)
        return 2.0 * positive * (1.0 - positive)

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
        if not rows or len(rows) != len(labels):
            raise ValueError("rows and labels must be non-empty and have equal length")
        self.root = self._grow(list(zip(rows, labels)), 0)

    def _grow(self, samples: list[tuple[Sequence[float], int]], depth: int) -> tuple:
        labels = [label for _, label in samples]
        probability = sum(labels) / len(labels)
        if (
            depth >= self.max_depth
            or len(samples) < self.min_samples
            or probability in (0.0, 1.0)
        ):
            return ("leaf", probability)

        best: tuple[float, int, float, list, list] | None = None
        feature_count = len(samples[0][0])
        for feature in range(feature_count):
            values = sorted({row[feature] for row, _ in samples})
            for left_value, right_value in zip(values, values[1:]):
                threshold = (left_value + right_value) / 2.0
                left = [item for item in samples if item[0][feature] <= threshold]
                right = [item for item in samples if item[0][feature] > threshold]
                if not left or not right:
                    continue
                score = (
                    len(left) * self._gini([label for _, label in left])
                    + len(right) * self._gini([label for _, label in right])
                ) / len(samples)
                if best is None or score < best[0]:
                    best = (score, feature, threshold, left, right)

        if best is None:
            return ("leaf", probability)
        _, feature, threshold, left, right = best
        return (
            "node",
            feature,
            threshold,
            self._grow(left, depth + 1),
            self._grow(right, depth + 1),
        )

    def predict_probability(self, row: Sequence[float]) -> float:
        if self.root is None:
            raise RuntimeError("classifier must be fitted before prediction")
        node = self.root
        while node[0] == "node":
            node = node[3] if row[node[1]] <= node[2] else node[4]
        return float(node[1])


def _feature_vector(
    page: int, cache: Sequence[int], last_seen: dict[int, int], frequencies: dict[int, int], now: int
) -> tuple[float, float, float]:
    return (
        float(now - last_seen.get(page, -1)),
        float(frequencies.get(page, 0)),
        float(page),
    )


def _future_distance(trace: Sequence[int], index: int, page: int) -> int:
    try:
        return trace.index(page, index + 1) - index
    except ValueError:
        return 10**9


def optimal_victim(cache: Sequence[int], trace: Sequence[int], index: int) -> int:
    return max(cache, key=lambda page: _future_distance(trace, index, page))


def train_learned_policy(trace: Sequence[int], capacity: int) -> DecisionTreeClassifier:
    rows: list[tuple[float, float, float]] = []
    labels: list[int] = []
    cache: list[int] = []
    last_seen: dict[int, int] = {}
    frequencies: dict[int, int] = {}
    for index, page in enumerate(trace):
        if page in cache:
            last_seen[page] = index
            frequencies[page] = frequencies.get(page, 0) + 1
            continue
        if len(cache) == capacity:
            victim = optimal_victim(cache, trace, index)
            for candidate in cache:
                rows.append(_feature_vector(candidate, cache, last_seen, frequencies, index))
                labels.append(int(candidate == victim))
            cache.remove(victim)
        cache.append(page)
        last_seen[page] = index
        frequencies[page] = frequencies.get(page, 0) + 1
    model = DecisionTreeClassifier()
    model.fit(rows, labels)
    return model


def simulate(
    trace: Sequence[int],
    capacity: int,
    policy: str,
    model: DecisionTreeClassifier | None = None,
    split_index: int | None = None,
) -> PolicyResult:
    if capacity < 1:
        raise ValueError("capacity must be positive")
    if policy == "learned" and model is None:
        raise ValueError("learned policy requires a fitted model")
    split_index = split_index if split_index is not None else len(trace) // 2
    cache: list[int] = []
    order: list[int] = []
    last_seen: dict[int, int] = {}
    frequencies: dict[int, int] = {}
    faults: list[int] = []

    for index, page in enumerate(trace):
        hit = page in cache
        faults.append(0 if hit else 1)
        if hit:
            if policy == "lru":
                order.remove(page)
                order.append(page)
            last_seen[page] = index
            frequencies[page] = frequencies.get(page, 0) + 1
            continue

        if len(cache) == capacity:
            if policy == "fifo":
                victim = order.pop(0)
            elif policy == "lru":
                victim = order.pop(0)
            elif policy == "optimal":
                victim = optimal_victim(cache, trace, index)
                order.remove(victim)
            elif policy == "learned":
                assert model is not None
                scored = [
                    (
                        model.predict_probability(
                            _feature_vector(candidate, cache, last_seen, frequencies, index)
                        ),
                        index - last_seen.get(candidate, -1),
                        candidate,
                    )
                    for candidate in cache
                ]
                victim = max(scored)[2]
                order.remove(victim)
            else:
                raise ValueError(f"unknown policy: {policy}")
            cache.remove(victim)

        cache.append(page)
        order.append(page)
        last_seen[page] = index
        frequencies[page] = frequencies.get(page, 0) + 1

    return PolicyResult(
        name=policy,
        hits=len(trace) - sum(faults),
        faults=sum(faults),
        total=len(trace),
        split_index=split_index,
        _faults=tuple(faults),
    )


def run_all_policies(
    trace: Sequence[int], capacity: int, training_trace: Sequence[int] | None = None
) -> dict[str, PolicyResult]:
    model = train_learned_policy(training_trace or trace, capacity)
    split = len(trace) // 2
    return {
        policy: simulate(trace, capacity, policy, model=model, split_index=split)
        for policy in ("fifo", "lru", "optimal", "learned")
    }
