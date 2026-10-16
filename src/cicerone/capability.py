"""Metaclass for optional-capability Protocols."""

from __future__ import annotations

from typing import Any, _ProtocolMeta


class CapabilityMeta(_ProtocolMeta):
    """``isinstance`` probes like ``callable(getattr(obj, name, None))``.

    Unlike the stock runtime check, it honours ``__getattr__`` and rejects non-callable values
    for methods, on every Python version.
    """

    def __instancecheck__(cls, instance: Any) -> bool:
        annotations = cls.__dict__.get("__annotations__", {})
        names = {name for name in (*vars(cls), *annotations) if not name.startswith("_")}
        for name in names:
            member = cls.__dict__.get(name)
            if callable(member) and not isinstance(member, property):
                if not callable(getattr(instance, name, None)):
                    return False
            elif not hasattr(instance, name):
                return False
        return True
