"""Validation-only adapters for external engineering callables.

The external computational functions are not edited to accept AutoIA's standard
one-vector model signature.  These adapters only unpack a vector of inputs,
insert fixed engineering constants, and select scalar/array outputs.  Their
``__autoia_prepare_dispatch__`` hook asks AutoIA to prepare the *external*
function itself, so NumPy/math operations inside the pre-existing code retain
local semantic dispatch without process-global monkeypatching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ExternalVectorAdapter:
    function: Callable
    argument_builder: Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]]
    output_selector: Callable[[Any], Any] = lambda value: value

    def __call__(self, p):
        args, kwargs = self.argument_builder(p)
        return self.output_selector(self.function(*args, **kwargs))

    def __autoia_prepare_dispatch__(self, traced_solve):
        # Import lazily so the externally authored source modules themselves
        # remain independent of AutoIA.
        from autoia.semantic.dispatch import prepare_model_for_dispatch

        prepared_function = prepare_model_for_dispatch(self.function, traced_solve)
        argument_builder = self.argument_builder
        output_selector = self.output_selector

        def prepared(p):
            args, kwargs = argument_builder(p)
            return output_selector(prepared_function(*args, **kwargs))

        return prepared
