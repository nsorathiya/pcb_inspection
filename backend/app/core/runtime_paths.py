from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_ROOT = REPOSITORY_ROOT / "runtime"


def resolve_runtime_root(runtime_root: Path) -> Path:
    """Resolve relative runtime roots against the repository, not the CWD."""
    expanded_root = runtime_root.expanduser()
    if not expanded_root.is_absolute():
        expanded_root = REPOSITORY_ROOT / expanded_root
    return expanded_root.resolve()


def default_runtime_root() -> Path:
    return DEFAULT_RUNTIME_ROOT


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    raw_uploads: Path
    previews: Path
    results: Path
    reports: Path
    temporary: Path

    @classmethod
    def from_root(cls, runtime_root: Path) -> "RuntimePaths":
        root = resolve_runtime_root(runtime_root)
        return cls(
            root=root,
            raw_uploads=root / "raw_uploads",
            previews=root / "previews",
            results=root / "results",
            reports=root / "reports",
            temporary=root / "tmp",
        )

    @property
    def directories(self) -> tuple[Path, ...]:
        return (
            self.root,
            self.raw_uploads,
            self.previews,
            self.results,
            self.reports,
            self.temporary,
        )

    def create_directories(self) -> None:
        """Create the runtime directory tree safely and idempotently."""
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)
