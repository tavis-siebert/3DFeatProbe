# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class MetricTracker:
    """Running stats for a scalar metric."""
    __slots__ = ("metric_name", "total", "count", "value")

    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
        self.value = 0.0  # most recent value

    def update(self, val: float, n: int = 1) -> None:
        self.value = float(val)
        self.total += val * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)

    def __str__(self) -> str:
        return f"{self.metric_name}: {self.value:.4f} (avg {self.average:.4f})"