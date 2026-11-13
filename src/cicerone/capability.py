"""Metaclass for optional-capability Protocols."""

from __future__ import annotations

from typing import Any, Generic, Protocol, _ProtocolMeta


class CapabilityMeta(_ProtocolMeta):
    """``isinstance`` probes like ``callable(getattr(obj, name, None))``.

    Unlike the stock runtime check, it honours ``__getattr__`` and rejects non-callable values
    for methods, on every Python version.
    """

    def __instancecheck__(cls, instance: Any) -> bool:
        for name, member in _protocol_members(cls).items():
            if callable(member) and not isinstance(member, property):
                if not callable(getattr(instance, name, None)):
                    return False
            elif not hasattr(instance, name):
                return False
        return True


def _protocol_members(cls: type) -> dict[str, Any]:
    # Inherited protocol members count too, so a protocol combining capabilities checks all of them.
    members: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        if base in (object, Protocol, Generic):
            continue
        for name in (*vars(base), *base.__dict__.get("__annotations__", {})):
            if not name.startswith("_"):
                members[name] = base.__dict__.get(name)
    return members
