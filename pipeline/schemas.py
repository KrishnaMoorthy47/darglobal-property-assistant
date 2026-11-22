"""Row schemas used by the pipeline and the app."""
from __future__ import annotations

from pydantic import BaseModel, field_validator


class Property(BaseModel):
    id: str
    source: str                   # darglobal | wasalt
    url: str
    name: str
    purpose: str | None = None    # sale | rent
    property_type: str | None = None
    city: str | None = None
    district: str | None = None
    country: str | None = None
    price: float | None = None
    currency: str | None = None
    price_usd: float | None = None
    rent_frequency: str | None = None
    bedrooms: float | None = None
    bedrooms_max: float | None = None
    bathrooms: float | None = None
    area_sqm: float | None = None
    status: str | None = None
    brand: str | None = None
    handover: str | None = None
    payment_plan: str | None = None
    furnishing: str | None = None
    image: str | None = None
    description: str = ""
    updated_at: str | None = None

    @field_validator("bedrooms", "bedrooms_max", "bathrooms", "area_sqm", "price", mode="before")
    @classmethod
    def _numeric(cls, v):
        if v in (None, "", "null"):
            return None
        try:
            return float(str(v).replace(",", ""))
        except ValueError:
            return None

    @field_validator("name", "description", "property_type", "city", "country", "status", "brand",
                     "handover", "payment_plan", mode="before")
    @classmethod
    def _stringify(cls, v):
        if isinstance(v, list):
            v = ", ".join(map(str, v))
        return v.strip() if isinstance(v, str) else v


class Chunk(BaseModel):
    id: str
    source: str
    doc_type: str
    title: str
    url: str
    text: str
    image: str | None = None
