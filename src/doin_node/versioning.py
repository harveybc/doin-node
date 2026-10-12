"""Resolve exact source revisions for the components used by a DOIN node."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse


_COMPONENTS: dict[str, tuple[str, tuple[Path, ...]]] = {
    "agent-multi": (
        "agent-multi",
        (
            Path.home() / "Documents" / "GitHub" / "agent-multi",
            Path.home() / "agent-multi",
        ),
    ),
    "doin-core": (
        "doin-core",
        (
            Path.home() / "Documents" / "GitHub" / "doin-core",
            Path.home() / "doin-core",
        ),
    ),
    "doin-node": (
        "doin-node",
        (
            Path.home() / "Documents" / "GitHub" / "doin-node",
            Path.home() / "doin-node",
        ),
    ),
    "doin-plugins": (
        "doin-plugins",
        (
            Path.home() / "Documents" / "GitHub" / "doin-plugins",
            Path.home() / "doin-plugins",
        ),
    ),
    "gym-fx": (
        "gym-fx",
        (
            Path.home() / "Documents" / "GitHub" / "gym-fx",
            Path.home() / "gym-fx",
        ),
    ),
    "predictor": (
        "predictor",
        (
            Path.home() / "Documents" / "GitHub" / "predictor",
            Path.home() / "predictor",
            Path("/var/tmp/i6n-island-sequence/code/predictor"),
        ),
    ),
    "trading-contracts": (
        "trading-contracts",
        (
            Path.home() / "Documents" / "GitHub" / "trading-contracts",
            Path.home() / "trading-contracts",
        ),
    ),
}

_RUNTIME_MODULES = {
    "doin-core": "doin_core",
    "doin-node": "doin_node",
    "doin-plugins": "doin_plugins",
    "predictor": "optimizer_plugins",
}


def _git_short_hash(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=path,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _editable_source(distribution_name: str) -> Path | None:
    """Return the source checkout recorded by an editable installation."""
    try:
        direct_url = importlib.metadata.distribution(distribution_name).read_text(
            "direct_url.json"
        )
        payload = json.loads(direct_url or "{}")
        parsed = urlparse(str(payload.get("url") or ""))
        if parsed.scheme != "file":
            return None
        source = Path(unquote(parsed.path)).resolve()
        return source if (source / ".git").exists() else None
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, OSError):
        return None


def _runtime_source(label: str) -> Path | None:
    """Return the Git worktree that supplies the module import at runtime."""
    module_name = _RUNTIME_MODULES.get(label)
    if module_name is None:
        return None
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or origin in {"built-in", "frozen"}:
        return None
    source = Path(origin).resolve()
    for parent in (source.parent, *source.parents):
        if (parent / ".git").exists():
            return parent
    return None


def compute_component_versions() -> dict[str, str]:
    """Return exact Git revisions for the packages participating in this run."""
    versions: dict[str, str] = {}
    for label, (distribution_name, candidates) in _COMPONENTS.items():
        source = _runtime_source(label) or _editable_source(distribution_name)
        revision = _git_short_hash(source) if source else None
        if revision is None:
            for candidate in candidates:
                if not (candidate / ".git").exists():
                    continue
                revision = _git_short_hash(candidate)
                if revision:
                    break
        if revision is not None:
            versions[label] = revision
            continue
        try:
            versions[label] = f"v{importlib.metadata.version(distribution_name)}"
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "?"
    return versions
