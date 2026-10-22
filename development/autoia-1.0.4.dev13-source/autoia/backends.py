"""Trusted arithmetic backends exposed by the stable AutoIA API."""

from .trusted.binary64 import RigorousBinary64Backend as _TrustedRigorousBinary64Backend
from .trusted.environment import Binary64EnvironmentError
from .exceptions import NumericalLimitError, InvalidInputError


class RigorousBinary64Backend(_TrustedRigorousBinary64Backend):
    """Public rigorous binary64 backend with stable typed environment failure."""

    def __init__(self, reduction_method="auto", vectorized_reduction_min_terms=16):
        try:
            super().__init__(reduction_method=reduction_method, vectorized_reduction_min_terms=vectorized_reduction_min_terms)
        except Binary64EnvironmentError as exc:
            raise NumericalLimitError(
                str(exc),
                context={"proof_layer": "binary64-environment"},
            ) from exc
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(str(exc), context={"configuration": "trusted reductions"}) from exc


__all__ = ["RigorousBinary64Backend"]
