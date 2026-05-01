from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable
import json
import re


FITS_EXTENSIONS = {".fit", ".fits", ".fts", ".new"}
RASTER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
IMAGE_EXTENSIONS = FITS_EXTENSIONS | RASTER_EXTENSIONS


@dataclass
class FrameInfo:
    path: str
    width: int | None = None
    height: int | None = None
    date_obs: str | None = None
    filter_name: str | None = None
    has_wcs: bool = False

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass
class PipelineSettings:
    index_dir: str = str(Path.home() / "astrometry-indexes" / "4200")
    scale_low: float | None = None
    scale_high: float | None = None
    hot_sigma: float = 8.0
    detect_sigma: float = 4.0
    min_detections: int = 3
    observatory_code: str = "500"


@dataclass
class SessionState:
    input_dir: str | None = None
    output_dir: str | None = None
    frames: list[FrameInfo] = field(default_factory=list)
    settings: PipelineSettings = field(default_factory=PipelineSettings)

    def frame_paths(self) -> list[Path]:
        return [Path(frame.path) for frame in self.frames]


def discover_fits_files(folder: str | Path) -> list[Path]:
    root = Path(folder).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    return natural_sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in FITS_EXTENSIONS)


def filter_image_files(paths: list[str] | tuple[str, ...] | list[Path]) -> list[Path]:
    return [Path(path).expanduser() for path in paths if Path(path).suffix.lower() in IMAGE_EXTENSIONS]


def natural_sort_key(path: str | Path) -> tuple[object, ...]:
    name = Path(path).name.lower()
    parts = re.split(r"(\d+)", name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def natural_sorted(paths: Iterable[str | Path]) -> list[Path]:
    return sorted((Path(path) for path in paths), key=natural_sort_key)


def save_session(state: SessionState, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    return output


def load_session(path: str | Path) -> SessionState:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    settings = PipelineSettings(**raw.get("settings", {}))
    frames = [FrameInfo(**item) for item in raw.get("frames", [])]
    return SessionState(
        input_dir=raw.get("input_dir"),
        output_dir=raw.get("output_dir"),
        frames=frames,
        settings=settings,
    )
