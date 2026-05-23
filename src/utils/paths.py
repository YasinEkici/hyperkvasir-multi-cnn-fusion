"""Project path helpers."""

from pathlib import Path


def project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def results_dir(exp_id: str) -> Path:
    path = project_root() / "results" / "runs" / exp_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_cache_dir() -> Path:
    path = project_root() / "results" / "feature_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def splits_dir(protocol: str = "hyperkvasir_official_5fold") -> Path:
    return project_root() / "data" / "splits" / protocol
