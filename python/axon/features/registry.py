"""The feature registry — the name a :class:`~axon.features.spec.FeatureSpec` writes down.

A spec that lives inside a model artifact has to survive being read back months
later by a process that has never seen the training script. It therefore refers to
transforms by **name**, not by function object, and this module is what turns a
name back into the one implementation of that transform.

Registration also pins the *shape* of the call: which arrays a feature consumes
(``inputs``) and which knobs it accepts (``params``). Both are derived from the
function signature rather than restated, because a registry that can disagree with
the code it indexes is worse than no registry — a spec would validate against the
declaration and then call something else.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np

FeatureFn = Callable[..., np.ndarray]


@dataclass(frozen=True)
class FeatureInfo:
    """One registered transform: its name, its callable, and its call shape."""

    name: str
    fn: FeatureFn
    #: Positional array arguments, in order. These are the binding names a spec uses.
    inputs: tuple[str, ...]
    #: Keyword-only tunables (window lengths, spans, …).
    params: tuple[str, ...]


_REGISTRY: dict[str, FeatureInfo] = {}


def register(name: str, *, inputs: tuple[str, ...]) -> Callable[[FeatureFn], FeatureFn]:
    """Register ``fn`` under ``name``, declaring its input arrays.

    The declared ``inputs`` must be exactly the function's positional parameters.
    Letting them drift would mean a spec binds ``price`` to a column while the
    function reads the array in a different slot — a silent feature swap that no
    downstream test can distinguish from a bad model.
    """

    def decorate(fn: FeatureFn) -> FeatureFn:
        sig = inspect.signature(fn)
        positional = tuple(
            p.name for p in sig.parameters.values() if p.kind is p.POSITIONAL_OR_KEYWORD
        )
        params = tuple(p.name for p in sig.parameters.values() if p.kind is p.KEYWORD_ONLY)
        if positional != tuple(inputs):
            raise TypeError(
                f"feature {name!r} declares inputs {tuple(inputs)} but its positional "
                f"parameters are {positional}; the declaration is what specs bind to"
            )
        if name in _REGISTRY:
            raise ValueError(f"feature {name!r} is already registered")
        _REGISTRY[name] = FeatureInfo(name=name, fn=fn, inputs=tuple(inputs), params=params)
        return fn

    return decorate


def feature_info(name: str) -> FeatureInfo:
    """Look up a registered feature, or raise naming what *is* registered."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownFeature(
            f"unknown feature {name!r}; registered features are {registered_features()}"
        ) from None


def registered_features() -> tuple[str, ...]:
    """Every registered feature name, sorted."""
    return tuple(sorted(_REGISTRY))


class FeatureError(ValueError):
    """Base for everything this package refuses to compute."""


class UnknownFeature(FeatureError):
    """A spec names a transform this build of the library does not have."""


__all__ = [
    "FeatureError",
    "FeatureFn",
    "FeatureInfo",
    "UnknownFeature",
    "feature_info",
    "register",
    "registered_features",
]
