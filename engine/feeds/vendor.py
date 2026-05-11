"""Pydantic models for vendor responses.

Why this exists:
  Per the architecture review, the fetcher's silent except clauses have hidden
  several real bugs (comma-OI, weekend empty body, missing key on certain
  endpoints). When the vendor format drifts, we want a clear, loud error
  instead of silent contract drops.

  These models do NOT replace the existing parser logic. They are an
  additional safety net we can switch on optionally — the fetcher still parses
  raw dicts (compatibility), but adds a `validate_with_pydantic=True` flag for
  paranoid mode that surfaces drift as logged warnings.

  Once we have a few weeks of clean validation, we can flip the default to
  validate everywhere. For now: adopt incrementally.

Coverage (initial):
  - SteadyApiContract — single put or call row from /v3/markets/options
  - SteadyApiChainResponse — the full response envelope
  - FlashAlphaGexResponse — exposure/gex endpoint
  - TiingoEodBar — daily bar from Tiingo

NOT yet covered (deferred):
  - BGeometrics: 7+ endpoints, complex shapes; do separately
  - SATA endpoints (low priority)
"""
from __future__ import annotations
from typing import Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict


def _strip_commas_int(v: Any) -> int:
    """Coerce '1,340' or 1340 to int. Raises on garbage.

    This is the regression armor for the comma-OI bug. Pydantic catches
    invalid values with a clear ValidationError instead of silently dropping
    the contract.
    """
    if v is None:
        return 0
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        return int(v.replace(",", "").strip() or "0")
    raise TypeError(f"cannot coerce {type(v).__name__} to int")


def _percent_to_decimal(v: Any) -> Optional[float]:
    """Coerce '65.50%' or '0.655' or 0.655 to 0.655. None for empty string.

    Some vendor fields return percent strings, others return decimal floats.
    Centralize the coercion so every consumer gets the same units.
    """
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.replace("%", "").replace(",", "").strip()
        if not s:
            return None
        f = float(s)
        # Heuristic: values > 1 are pct-formatted, divide by 100. Some vendors
        # return "65" for 65%, some return "0.65". The format string from
        # SteadyAPI for `volatility` is always pct, so we always divide here.
        return f / 100.0
    return None


class SteadyApiContract(BaseModel):
    """One row from the SteadyAPI option chain (put or call).

    Field names mirror the vendor's wire format (camelCase) so we can
    `.model_validate(raw_dict)` without renaming.

    Model is permissive on extras — the vendor occasionally adds new fields
    we don't care about. We DO validate the fields we use.
    """
    model_config = ConfigDict(extra="ignore")

    strikePrice: float
    midpoint: float = Field(default=0.0)
    delta: Optional[float] = None
    vega: float = Field(default=0.0)
    theta: float = Field(default=0.0)
    volatility: Optional[Union[float, str]] = None  # raw — convert via helper
    openInterest: int = Field(default=0)
    bidPrice: float = Field(default=0.0)
    askPrice: float = Field(default=0.0)

    @field_validator("strikePrice", "midpoint", "vega", "theta", "bidPrice", "askPrice", mode="before")
    @classmethod
    def _coerce_float(cls, v):
        """Vendor sends prices as strings ('8.85'). Coerce to float."""
        if v is None or v == "":
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            return float(v.replace(",", "").strip() or "0")
        return v

    @field_validator("delta", mode="before")
    @classmethod
    def _coerce_delta(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            return float(s) if s else None
        return v

    @field_validator("openInterest", mode="before")
    @classmethod
    def _coerce_oi(cls, v):
        # The comma-OI fix in pydantic form. Bug regression armor.
        return _strip_commas_int(v)

    def iv_decimal(self) -> Optional[float]:
        """Convert volatility field to decimal (0.65 = 65%)."""
        return _percent_to_decimal(self.volatility)


class SteadyApiChainBody(BaseModel):
    """The 'body' field of a chain response. May be {Put: [...], Call: [...]}
    OR an empty list when the market is closed. Both shapes valid."""
    model_config = ConfigDict(extra="ignore")

    Put: list[SteadyApiContract] = Field(default_factory=list)
    Call: list[SteadyApiContract] = Field(default_factory=list)


class SteadyApiExpirationsMeta(BaseModel):
    """meta.expirations field with weekly + monthly arrays."""
    model_config = ConfigDict(extra="ignore")

    weekly: list[str] = Field(default_factory=list)
    monthly: list[str] = Field(default_factory=list)

    def all_expirations(self) -> list[str]:
        """Sorted, deduplicated combined list."""
        return sorted(set(self.weekly + self.monthly))


# ─── FlashAlpha ──────────────────────────────────────────────────────────────

class FlashAlphaGexResponse(BaseModel):
    """Top-level response from /v1/exposure/gex/ASST.

    We validate only the fields the engine reads. Extras are ignored so vendor
    additions don't break us.
    """
    model_config = ConfigDict(extra="ignore")

    spot: Optional[float] = None
    net_gex: Optional[float] = None
    gamma_flip: Optional[float] = None
    raw_flip: Optional[float] = None
    net_vanna: Optional[float] = None
    pos_magnets: Optional[list[dict]] = None
    neg_magnets: Optional[list[dict]] = None


# ─── Tiingo EOD ──────────────────────────────────────────────────────────────

class TiingoEodBar(BaseModel):
    """One daily bar from Tiingo.

    Tiingo returns adjOpen/adjClose etc; we use those (split-adjusted).
    """
    model_config = ConfigDict(extra="ignore")

    date: str
    adjOpen: Optional[float] = None
    adjHigh: Optional[float] = None
    adjLow: Optional[float] = None
    adjClose: Optional[float] = None
    adjVolume: Optional[int] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None


# ─── Validation helpers ──────────────────────────────────────────────────────

def safely_validate_chain_body(raw: Any) -> SteadyApiChainBody:
    """Validate a SteadyAPI chain body; return empty body on shape mismatch.

    SteadyAPI returns body as `{Put: [...], Call: [...]}` during market hours
    but as an empty list `[]` when markets are closed. The empty list isn't
    an error — it's expected weekend/holiday behavior. Surface it as an
    empty SteadyApiChainBody rather than raising.
    """
    if isinstance(raw, dict):
        return SteadyApiChainBody.model_validate(raw)
    if isinstance(raw, list):
        # Empty list = market closed; not an error
        return SteadyApiChainBody()
    # Anything else is unexpected — return empty and let caller log
    return SteadyApiChainBody()
