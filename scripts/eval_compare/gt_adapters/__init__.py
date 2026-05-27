"""Registry of dataset GT adapters. Each module exposes ``list_sequences`` and
``load_sequence`` returning a ``GTSequence`` (see base.py)."""

from . import h2o, egoverse, oakink2, taco
from .base import GTSequence

ADAPTERS = {
    "h2o": h2o,
    "egoverse": egoverse,
    "oakink2": oakink2,
    "taco": taco,
}


def get_adapter(name: str):
    if name not in ADAPTERS:
        raise KeyError(f"unknown dataset '{name}'; choose from {list(ADAPTERS)}")
    return ADAPTERS[name]
