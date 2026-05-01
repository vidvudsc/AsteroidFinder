"""Astronomical image loading, solving, alignment, and tracking tools."""

from .alignment import AlignedFrame, align_images, stack_images
from .calibration import (
    CalibrationResult,
    apply_hot_pixel_mask,
    build_persistent_hot_pixel_mask,
    calibrate_image,
    calibrate_images,
    calibrate_images_with_persistent_hot_pixels,
    detect_isolated_hot_pixels,
    make_master_frame,
    remove_hot_pixels,
)
from .detection import Source, detect_sources
from .diagnostics import plot_track_diagnostics
from .doctor import DoctorCheck, install_astrometry_indexes, recommend_index_series, run_doctor
from .io import AstroImage, load_image, save_fits, save_jpeg
from .known_objects import (
    KnownObject,
    KnownObjectPhotometry,
    forced_photometry_for_known_objects,
    query_known_objects_for_frames,
    query_known_objects_in_frame,
    write_mpc_observations,
)
from .platesolve import PlateSolution, solve_image
from .photometry import Photometry, aperture_photometry
from .tracking import Track, TrackDetection, track_aligned_frames, track_moving_objects
from .workflow import AsteroidWorkflowResult, run_asteroid_workflow

__all__ = [
    "AlignedFrame",
    "AstroImage",
    "AsteroidWorkflowResult",
    "CalibrationResult",
    "DoctorCheck",
    "PlateSolution",
    "Photometry",
    "KnownObject",
    "KnownObjectPhotometry",
    "Source",
    "Track",
    "TrackDetection",
    "align_images",
    "aperture_photometry",
    "apply_hot_pixel_mask",
    "build_persistent_hot_pixel_mask",
    "calibrate_image",
    "calibrate_images",
    "calibrate_images_with_persistent_hot_pixels",
    "detect_isolated_hot_pixels",
    "forced_photometry_for_known_objects",
    "detect_sources",
    "load_image",
    "make_master_frame",
    "install_astrometry_indexes",
    "recommend_index_series",
    "remove_hot_pixels",
    "query_known_objects_for_frames",
    "query_known_objects_in_frame",
    "plot_track_diagnostics",
    "write_mpc_observations",
    "run_asteroid_workflow",
    "run_doctor",
    "save_fits",
    "save_jpeg",
    "solve_image",
    "stack_images",
    "track_aligned_frames",
    "track_moving_objects",
]
