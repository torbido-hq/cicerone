"""Parse prometheus_client text exposition in tests."""

from __future__ import annotations

from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families


def metric_samples(body: str, name: str) -> list[tuple[dict[str, str], float]]:
    out: list[tuple[dict[str, str], float]] = []
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if sample.name != name:
                continue
            out.append((dict(sample.labels), sample.value))
    return out


def metric_value(body: str, name: str, labels: dict[str, str] | None = None) -> float:
    total = 0.0
    for sample_labels, value in metric_samples(body, name):
        if labels is not None and sample_labels != labels:
            continue
        total += value
    return total


def registry_metric_value(name: str, labels: dict[str, str] | None = None) -> float:
    return metric_value(generate_latest().decode(), name, labels)
