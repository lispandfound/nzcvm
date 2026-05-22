"""Decorator for constructing :class:`~nzcvm.layers.core.Layer` subclasses
from plain functions.

A *functional layer* is a Layer whose logic is expressed as a single Python
function.  The decorator inspects the function's keyword parameters and
generates:

1. A :class:`~nzcvm.config.layers.core.LayerConfig` dataclass whose fields
   correspond to those keyword parameters.
2. A :class:`~nzcvm.layers.core.Layer` subclass whose ``__init__`` accepts
   ``(next_layer=None, **config_kwargs)`` and whose ``__call__`` delegates to
   the wrapped function.

The decorated function must accept the following positional arguments:

* ``grid`` — the current :class:`~nzcvm.grids.Grid` chunk
* ``model_range`` — the active :class:`~nzcvm.model.ModelRange`

and the following keyword argument:

* ``next_layer`` — the downstream :class:`~nzcvm.layers.core.Layer` (or
  ``None`` for terminal layers)

All other keyword parameters become config fields on the generated
``LayerConfig`` subclass.

Example
-------
::

    from nzcvm.layers.functional import functional_layer
    from nzcvm.grids import Grid
    from nzcvm.model import ModelRange

    @functional_layer
    def scale(
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
        *,
        next_layer,
        factor: float = 1.0,
    ):
        qualities = next_layer(grid, model_range=model_range)
        qualities["vp"] *= factor
        return qualities

    # Instantiate: keyword args populate the generated ScaleConfig.
    layer = scale(factor=2.0, next_layer=some_terminal)

"""

from __future__ import annotations

import inspect
from dataclasses import field, make_dataclass
from typing import Any, Callable, get_type_hints

from mashumaro.core.meta.mixin import compile_mixin_packer, compile_mixin_unpacker

from nzcvm.config.layers.core import LayerConfig
from nzcvm.grids import Grid
from nzcvm.layers.core import Layer
from nzcvm.model import ModelRange
from nzcvm.qualities import Qualities

_RUNTIME_PARAMS = frozenset({"grid", "model_range", "next_layer", "return"})


def _recompile_mashumaro_codecs(cls: type) -> None:
    """Recompile mashumaro packer/unpacker codecs for *cls*.

    On Python 3.14, :func:`dataclasses.make_dataclass` sets ``__annotate__``
    *after* :func:`types.new_class` returns.  Mashumaro's
    ``__init_subclass__`` hook fires during class creation — before
    annotations are accessible — so the generated codecs are empty.

    This function replays the same ancestor walk that ``__init_subclass__``
    performs, picking up every mixin's builder params (dict, JSON, TOML,
    YAML, …) so that all serialisation formats work correctly.
    """
    # Walk from the most-base ancestor down to the direct parent, mirroring
    # the order mashumaro's own __init_subclass__ uses.
    for ancestor in reversed(cls.__mro__[1:]):
        for attr_name in vars(ancestor):
            if attr_name.endswith("__mashumaro_builder_params"):
                bp = getattr(ancestor, attr_name)
                compile_mixin_unpacker(cls, **bp["unpacker"])
                compile_mixin_packer(cls, **bp["packer"])


def functional_layer(func: Callable[..., Qualities]) -> type[Layer]:
    """Derive a :class:`~nzcvm.layers.core.Layer` subclass from *func*.

    Parameters
    ----------
    func :
        A callable with signature
        ``(grid, model_range, *, next_layer, **config_params) -> Qualities``.

    Returns
    -------
    type[Layer]
        A new, registered Layer subclass whose name matches *func.__name__*.
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    config_fields: list = []
    param_names: list[str] = []

    for name, param in sig.parameters.items():
        if name in _RUNTIME_PARAMS:
            continue
        ann = hints.get(name, Any)
        if param.default is inspect.Parameter.empty:
            config_fields.append((name, ann))
        else:
            config_fields.append((name, ann, field(default=param.default)))
        param_names.append(name)

    type_tag = func.__name__
    # Discriminator field — str so mashumaro serialises it on all Python versions.
    config_fields.append(("type", str, field(default=type_tag)))

    config_name = func.__name__.title().replace("_", "") + "Config"
    ConfigCls: type[LayerConfig] = make_dataclass(  # type: ignore[assignment]
        config_name,
        config_fields,
        bases=(LayerConfig,),
    )

    _recompile_mashumaro_codecs(ConfigCls)

    _captured_param_names = list(param_names)

    def _init(self: Any, next_layer: Layer | None = None, **kwargs: Any) -> None:
        config = ConfigCls(**kwargs)
        Layer.__init__(self, config, next_layer)  # type: ignore[arg-type]
        for n in _captured_param_names:
            setattr(self, n, getattr(config, n))

    def _call(
        self: Any,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        params = {n: getattr(self, n) for n in _captured_param_names}
        return func(grid, model_range, next_layer=self.next_layer, **params)

    LayerCls: type[Layer] = type(  # type: ignore[assignment]
        func.__name__,
        (Layer,),
        {
            "__init__": _init,
            "__call__": _call,
            "__doc__": func.__doc__,
            "config_cls": ConfigCls,
        },
        config_cls=ConfigCls,
    )

    return LayerCls
