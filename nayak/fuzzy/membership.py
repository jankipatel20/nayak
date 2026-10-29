"""Vectorized fuzzy membership functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Union
import numpy as np


class MembershipFunction(ABC):
    """Abstract base class for fuzzy membership functions."""

    @abstractmethod
    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute membership degree in [0, 1] for scalar or array inputs."""
        pass

    @abstractmethod
    def breakpoints(self) -> Sequence[float]:
        """Return the defining knot points of the membership function."""
        pass


class TriangularMF(MembershipFunction):
    """Triangular membership function defined by vertices (a, b, c).

    Degree is 0 for x <= a, ramps to 1 at b, ramps down to 0 at c, and 0 for x >= c.
    """

    def __init__(self, a: float, b: float, c: float) -> None:
        if not (a <= b <= c):
            raise ValueError(f"Triangular parameters must satisfy a <= b <= c, got ({a}, {b}, {c})")
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate triangular membership vectorized."""
        is_scalar = np.isscalar(x)
        arr = np.asarray(x, dtype=float)

        y = np.zeros_like(arr)

        # Left slope: a < x < b
        if self.b > self.a:
            left_mask = (arr > self.a) & (arr < self.b)
            y[left_mask] = (arr[left_mask] - self.a) / (self.b - self.a)

        # Peak: x == b
        y[arr == self.b] = 1.0

        # Right slope: b < x < c
        if self.c > self.b:
            right_mask = (arr > self.b) & (arr < self.c)
            y[right_mask] = (self.c - arr[right_mask]) / (self.c - self.b)

        return float(y.item()) if is_scalar else y

    def breakpoints(self) -> Sequence[float]:
        return (self.a, self.b, self.c)

    def __repr__(self) -> str:
        return f"TriangularMF({self.a:.2f}, {self.b:.2f}, {self.c:.2f})"


class TrapezoidalMF(MembershipFunction):
    """Trapezoidal membership function defined by vertices (a, b, c, d).

    Supports left shoulders (a == b) and right shoulders (c == d).
    """

    def __init__(self, a: float, b: float, c: float, d: float) -> None:
        if not (a <= b <= c <= d):
            raise ValueError(f"Trapezoidal parameters must satisfy a <= b <= c <= d, got ({a}, {b}, {c}, {d})")
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate trapezoidal membership vectorized."""
        is_scalar = np.isscalar(x)
        arr = np.asarray(x, dtype=float)

        y = np.zeros_like(arr)

        # Left slope: a < x < b
        if self.b > self.a:
            left_mask = (arr > self.a) & (arr < self.b)
            y[left_mask] = (arr[left_mask] - self.a) / (self.b - self.a)
        elif self.a == self.b:
            # Left shoulder: 1 for x <= b
            y[arr <= self.b] = 1.0

        # Plateau: b <= x <= c
        plateau_mask = (arr >= self.b) & (arr <= self.c)
        y[plateau_mask] = 1.0

        # Right slope: c < x < d
        if self.d > self.c:
            right_mask = (arr > self.c) & (arr < self.d)
            y[right_mask] = (self.d - arr[right_mask]) / (self.d - self.c)
        elif self.c == self.d:
            # Right shoulder: 1 for x >= c
            y[arr >= self.c] = 1.0

        return float(y.item()) if is_scalar else y

    def breakpoints(self) -> Sequence[float]:
        return (self.a, self.b, self.c, self.d)

    def __repr__(self) -> str:
        return f"TrapezoidalMF({self.a:.2f}, {self.b:.2f}, {self.c:.2f}, {self.d:.2f})"
