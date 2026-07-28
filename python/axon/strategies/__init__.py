"""``axon.strategies`` — real strategies, taken up the validation ladder.

``docs/07`` defines the ladder; ``axon.features``, ``axon.models`` and
``axon.parity`` build its rungs. This package is what climbs them: one honest,
small, real strategy plus everything needed to reproduce the claim that it did.

* :mod:`axon.strategies.data` — real Hyperliquid candles, cached, with a committed
  fixture so the default test gate is offline.
* :mod:`axon.strategies.labels` — the forward-return label and the purged
  walk-forward split. Deliberately not registered as features.
* :mod:`axon.strategies.perp_bar` — :data:`PERP_BAR_V1`, the versioned feature
  spec, and :class:`PerpBar`, the strategy that serves it.
* :mod:`axon.strategies.baseline` — the floor: a finite-window z-score rule with **no
  artifact at all**, so a failure that reaches the venue through it is the plumbing
  rather than the model.
* :mod:`axon.strategies.zoo` — the width: one shared :data:`~axon.features.BAR_M1_V1`
  recipe fitted by three model families and taken through all four gates (ADR-0032).
  Imported by path rather than re-exported here, because it pulls the export path in.
* :mod:`axon.strategies.training` — candles → features → labels → walk-forward →
  artifact → the three gates, in one call (``climb``).
* :mod:`axon.strategies.shadow` — rung 3: the serving path driven forward over a bar
  feed, its would-be targets on a real signal ring, and a continuous diff against the
  offline recompute of the bars it was shown (ADR-0029).
* :mod:`axon.strategies.jobs` — the same work expressed as an
  :mod:`axon.compute` job, for the sweep this box is too small for.

Run the whole thing::

    python -m axon.strategies --coins BTC ETH

which prints the walk-forward numbers and all three gate reports. It reads the
committed fixture unless ``--cache`` is given, and it never touches the network
unless ``AXON_ALLOW_NETWORK=1`` says it may.

The heavy imports live in the submodules, not here: :mod:`axon.strategies.training`
needs XGBoost and :mod:`axon.strategies.jobs` needs the compute client, and
importing this package must not require either.

What a run of this package is allowed to claim is the subject of ADR-0022, and the
short version is worth repeating here: passing these gates says the live path
computes what research computed. It does not say the strategy makes money.
"""

from __future__ import annotations

from axon.strategies.baseline import (
    BASELINE_Z_V1,
    NO_MODEL_VERSION,
    Baseline,
    BaselineParams,
    evaluate_baseline,
    warmup_bars,
    warmup_minutes,
)
from axon.strategies.data import (
    ALLOW_NETWORK_ENV,
    Candles,
    DataError,
    fixture_candles,
    fixture_coins,
    load_candles,
)
from axon.strategies.labels import (
    Fold,
    direction_label,
    forward_log_return,
    purged_walk_forward,
)
from axon.strategies.perp_bar import (
    LABEL_HORIZON_BARS,
    PERP_BAR_V1,
    SERVING_BUFFER_BARS,
    PerpBar,
    PerpBarParams,
)

__all__ = [
    "ALLOW_NETWORK_ENV",
    "BASELINE_Z_V1",
    "LABEL_HORIZON_BARS",
    "NO_MODEL_VERSION",
    "PERP_BAR_V1",
    "SERVING_BUFFER_BARS",
    "Baseline",
    "BaselineParams",
    "Candles",
    "DataError",
    "Fold",
    "PerpBar",
    "PerpBarParams",
    "direction_label",
    "evaluate_baseline",
    "fixture_candles",
    "fixture_coins",
    "forward_log_return",
    "load_candles",
    "purged_walk_forward",
    "warmup_bars",
    "warmup_minutes",
]
