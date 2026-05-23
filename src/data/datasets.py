"""Dataset definitions for HyperKvasir experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from src.data.manifests import read_manifest_csv


class HyperKvasirImageDataset(Dataset):
    """Image dataset backed by a manifest CSV or manifest rows."""

    def __init__(
        self,
        manifest: str | Path | list[dict],
        *,
        transform: Callable | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.rows = read_manifest_csv(manifest) if isinstance(manifest, (str, Path)) else manifest
        self.transform = transform
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[object, int, dict]:
        row = self.rows[index]
        image_path = Path(str(row["path"]))
        if not image_path.is_absolute():
            image_path = self.project_root / image_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        label = int(row["label"])
        return image, label, row
