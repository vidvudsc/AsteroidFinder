"""Astronomical image loading, solving, alignment, and tracking tools."""

from .alignment import AlignedFrame, align_images, stack_images
from .astrometry_quality import (
    AstrometryMatch,
    AstrometryQualityResult,
    GaiaReferenceSource,
    measure_gaia_astrometry_quality,
    query_gaia_sources_for_frame,
    run_gaia_astrometry_quality,
    write_astrometry_quality_csv,
    write_wcs_offset_corrected_fits,
    write_wcs_offset_corrections,
)
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
from .diagnostics import (
    plot_track_diagnostics,
    write_full_frame_tracks_png,
    write_track_cutout_gifs,
    write_track_diagnostic_outputs,
)
from .doctor import DoctorCheck, install_astrometry_indexes, recommend_index_series, run_doctor
from .ephemeris import MpcEphemerisPrediction, query_mpc_ephemeris_for_frames, write_mpc_ephemeris_csv
from .io import AstroImage, load_image, save_fits, save_jpeg
from .known_objects import (
    KnownObject,
    KnownObjectPhotometry,
    forced_photometry_for_known_objects,
    query_known_objects_for_frames,
    query_known_objects_in_frame,
    write_mpc_observations,
)
from .mpc import (
    DetectedTrackObservation,
    measured_observations_from_tracks,
    write_detected_track_mpc,
    write_detected_track_observations_csv,
)
from .outputs import OutputLayout, output_layout
from .platesolve import PlateSolution, solve_image, write_plate_solutions_csv
from .photometry import Photometry, aperture_photometry
from .tracking import Track, TrackDetection, track_aligned_frames, track_moving_objects
from .workflow import AsteroidWorkflowResult, run_asteroid_workflow

__all__ = [
    "AlignedFrame",
    "AstroImage",
    "AsteroidWorkflowResult",
    "AstrometryMatch",
    "AstrometryQualityResult",
    "CalibrationResult",
    "DoctorCheck",
    "DetectedTrackObservation",
    "GaiaReferenceSource",
    "PlateSolution",
    "Photometry",
    "KnownObject",
    "KnownObjectPhotometry",
    "MpcEphemerisPrediction",
    "OutputLayout",
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
    "measured_observations_from_tracks",
    "measure_gaia_astrometry_quality",
    "install_astrometry_indexes",
    "recommend_index_series",
    "remove_hot_pixels",
    "query_known_objects_for_frames",
    "query_gaia_sources_for_frame",
    "query_known_objects_in_frame",
    "query_mpc_ephemeris_for_frames",
    "plot_track_diagnostics",
    "write_mpc_observations",
    "write_mpc_ephemeris_csv",
    "write_detected_track_mpc",
    "write_detected_track_observations_csv",
    "run_asteroid_workflow",
    "run_doctor",
    "run_gaia_astrometry_quality",
    "save_fits",
    "save_jpeg",
    "solve_image",
    "stack_images",
    "track_aligned_frames",
    "track_moving_objects",
    "write_astrometry_quality_csv",
    "write_wcs_offset_corrected_fits",
    "write_wcs_offset_corrections",
    "write_full_frame_tracks_png",
    "write_plate_solutions_csv",
    "write_track_cutout_gifs",
    "write_track_diagnostic_outputs",
    "output_layout",
]
