from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Endpoint:
    """A target API endpoint for benchmarking."""

    label: str
    url: str
    api_key: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "url": self.url}


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


@dataclass
class Model:
    """Representation of an LLM model tailored for benchmarking.

    Keeps a small set of commonly useful attributes (size, parameter_size,
    quantization, family, context length) and preserves the raw `meta`.
    Provider-specific fields (size, family, etc.) will be None for providers
    that don't expose them (e.g. OpenAI-compatible endpoints).
    """

    name: str
    id: Optional[str] = None
    modified_at: Optional[datetime] = None
    size: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization_level: Optional[str] = None
    family: Optional[str] = None
    context_length: Optional[int] = None
    capabilities: Optional[List[str]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "Model":
        # Accept Pydantic/BaseModel-like objects with .dict()
        if hasattr(data, "dict") and callable(getattr(data, "dict")):
            try:
                data = data.dict()
            except Exception:
                data = dict(data)

        if isinstance(data, str):
            return cls(name=data)
        if not isinstance(data, dict):
            # Try attribute access for common fields
            name = getattr(data, "name", None) or getattr(data, "model", None)
            if name:
                return cls(name=str(name))
            return cls(name=str(data))

        name = data.get("name") or data.get("model") or data.get("id") or str(data)
        modified = _parse_datetime(data.get("modified_at") or data.get("modifiedAt") or data.get("modified"))

        # Details can be nested under 'details'
        details = data.get("details") or {}

        return cls(
            name=name,
            id=data.get("id") or data.get("model"),
            modified_at=modified,
            size=data.get("size") or details.get("size"),
            parameter_size=details.get("parameter_size") or details.get("parameterSize") or data.get("parameter_size"),
            quantization_level=details.get("quantization_level") or details.get("quantizationLevel") or data.get("quantization_level"),
            family=details.get("family") or data.get("family"),
            context_length=data.get("context_length") or data.get("contextLength") or details.get("context_length"),
            capabilities=data.get("capabilities") or data.get("capability") or [],
            meta=dict(data),
        )

    @classmethod
    def from_openai(cls, data: Any) -> "Model":
        """Create a Model from an openai.types.Model object (id, created, owned_by)."""
        model_id = getattr(data, "id", None) or str(data)
        created = getattr(data, "created", None)
        modified = None
        if created is not None:
            try:
                modified = datetime.fromtimestamp(int(created))
            except Exception:
                pass
        return cls(
            name=model_id,
            id=model_id,
            modified_at=modified,
            meta={"owned_by": getattr(data, "owned_by", None)},
        )

    def to_dict(self) -> Dict[str, Any]:
        result = dict(self.meta) if self.meta else {}
        result["name"] = self.name
        if self.id:
            result["id"] = self.id
        if self.modified_at:
            result["modified_at"] = self.modified_at.isoformat()
        if self.size:
            result["size"] = self.size
        if self.parameter_size:
            result["parameter_size"] = self.parameter_size
        if self.quantization_level:
            result["quantization_level"] = self.quantization_level
        if self.family:
            result["family"] = self.family
        if self.context_length:
            result["context_length"] = self.context_length
        if self.capabilities:
            result["capabilities"] = self.capabilities
        return result

