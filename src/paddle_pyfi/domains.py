from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DomainAdapter:
    name: str
    document_types: list[str]
    tasks: list[str]
    output_schema: dict[str, Any]
    forbidden: list[str]
    system_instructions: str
    evaluation_notes: list[str]


def _domain_package_path(name: str) -> resources.abc.Traversable:
    return resources.files("paddle_pyfi.domains_data").joinpath(f"{name}.yaml")


def available_domains() -> list[str]:
    root = resources.files("paddle_pyfi.domains_data")
    return sorted(path.name.removesuffix(".yaml") for path in root.iterdir() if path.name.endswith(".yaml"))


def load_domain(name_or_path: str) -> DomainAdapter:
    path = Path(name_or_path)
    if path.exists():
        raw = path.read_text(encoding="utf-8")
    else:
        resource = _domain_package_path(name_or_path)
        if not resource.is_file():
            known = ", ".join(available_domains())
            raise ValueError(f"Unknown domain '{name_or_path}'. Available domains: {known}")
        raw = resource.read_text(encoding="utf-8")

    data = yaml.safe_load(raw) or {}
    required = {"name", "document_types", "tasks", "output_schema", "forbidden", "system_instructions"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Domain adapter is missing required keys: {', '.join(missing)}")

    return DomainAdapter(
        name=str(data["name"]),
        document_types=list(data.get("document_types", [])),
        tasks=list(data.get("tasks", [])),
        output_schema=dict(data.get("output_schema", {})),
        forbidden=list(data.get("forbidden", [])),
        system_instructions=str(data.get("system_instructions", "")),
        evaluation_notes=list(data.get("evaluation_notes", [])),
    )
