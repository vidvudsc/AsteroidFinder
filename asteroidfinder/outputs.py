from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputLayout:
    """Canonical output paths for one AsteroidFinder run."""

    root: Path

    @classmethod
    def from_dir(cls, output_dir: str | Path) -> "OutputLayout":
        return cls(Path(output_dir))

    @property
    def calibrated_dir(self) -> Path:
        return self.root / "calibrated"

    @property
    def aligned_dir(self) -> Path:
        return self.root / "aligned"

    @property
    def solved_dir(self) -> Path:
        return self.root / "solved"

    @property
    def difference_dir(self) -> Path:
        return self.root / "difference"

    @property
    def qa_dir(self) -> Path:
        return self.root / "qa"

    @property
    def hot_pixel_qa_dir(self) -> Path:
        return self.qa_dir / "hot_pixels"

    @property
    def alignment_qa_csv(self) -> Path:
        return self.qa_dir / "alignment.csv"

    @property
    def astrometry_qa_dir(self) -> Path:
        return self.qa_dir / "astrometry"

    @property
    def plate_solve_csv(self) -> Path:
        return self.qa_dir / "plate_solve.csv"

    @property
    def detections_dir(self) -> Path:
        return self.root / "detections"

    @property
    def tracks_csv(self) -> Path:
        return self.detections_dir / "tracks.csv"

    @property
    def track_diagnostics_dir(self) -> Path:
        return self.detections_dir / "diagnostics"

    @property
    def known_dir(self) -> Path:
        return self.root / "known"

    @property
    def known_objects_csv(self) -> Path:
        return self.known_dir / "known_objects.csv"

    @property
    def target_ephemeris_csv(self) -> Path:
        return self.known_dir / "target_ephemeris.csv"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    @property
    def detected_track_mpc(self) -> Path:
        return self.export_dir / "detected_track_mpc.txt"

    @property
    def detected_track_ades(self) -> Path:
        return self.export_dir / "detected_track_ades.psv"

    @property
    def detected_track_observations_csv(self) -> Path:
        return self.export_dir / "detected_track_observations.csv"

    @property
    def submission_mpc(self) -> Path:
        return self.export_dir / "submission_mpc.txt"

    @property
    def submission_ades(self) -> Path:
        return self.export_dir / "submission_ades.psv"

    @property
    def submission_observations_csv(self) -> Path:
        return self.export_dir / "submission_observations.csv"


def output_layout(output_dir: str | Path) -> OutputLayout:
    return OutputLayout.from_dir(output_dir)
