#!/usr/bin/env python3
"""Galaxy naming utilities shared across analysis scripts."""

from __future__ import annotations

import re


def canonicalize_galaxy_name(name: str) -> str:
    """
    Canonical galaxy identifier for cross-dataset joins.

    Rules:
    - uppercase
    - strip outer whitespace
    - collapse internal whitespace
    - for non-WALLABY names, remove spaces entirely so
      "NGC 2403" == "NGC2403" and "UGC 89" == "UGC89"
    - keep WALLABY identifiers with single spaces (display-friendly)
    """
    s = str(name).strip().upper()
    s = re.sub(r"\s+", " ", s)
    if s.startswith("WALLABY"):
        return s
    return s.replace(" ", "")


__all__ = ["canonicalize_galaxy_name"]

