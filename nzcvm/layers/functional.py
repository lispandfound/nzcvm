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

import dataclasses
import inspect
from dataclasses import field
from typing import Any, Callable, get_type_hints

from nzcvm.config.layers.core import LayerConfig
from nzcvm.grids import Grid
from nzcvm.layers.core import Layer
from nzcvm.model import ModelRange
from nzcvm.qualities import Qualities

_RUNTIME_PARAMS = frozenset({"grid", "model_range", "next_layer", "return"})


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

    # Build the config class using type() + @dataclass rather than make_dataclass.
    # On Python 3.14, make_dataclass uses annotationlib and sets __annotate__
    # *after* types.new_class() returns, so mashumaro's __init_subclass__ hook
    # fires before annotations are accessible — producing an empty codec.
    # Using type() with __annotations__ already in the namespace ensures mashumaro
    # sees the fields at __init_subclass__ time on every Python version.
    ns: dict[str, Any] = {"__annotations__": {}}
    for name, ann, *rest in config_fields:
        ns["__annotations__"][name] = ann
        if rest:
            ns[name] = rest[0]  # field() default/factory descriptor
    # Discriminator field — str so mashumaro serialises it on all Python versions.
    ns["__annotations__"]["type"] = str
    ns["type"] = field(default=type_tag)

    config_name = func.__name__.title().replace("_", "") + "Config"
    ConfigCls: type[LayerConfig] = dataclasses.dataclass(  # type: ignore[assignment]
        type(config_name, (LayerConfig,), ns)
    )

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
