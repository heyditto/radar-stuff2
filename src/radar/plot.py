"""Figure for the gap-fill experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radar.grid import DBZ_MAX, DBZ_MIN, ReflectivityGrid


def reflectivity_cmap():
    """NWS-style reflectivity colors when cmweather is installed, else turbo."""
    try:
        import cmweather

        return cmweather.cm.NWSRef
    except Exception:
        return plt.get_cmap("turbo")


def _reflectivity(ax, grid: ReflectivityGrid, field: np.ndarray, title: str, cmap) -> None:
    image = ax.imshow(
        np.ma.masked_invalid(field),
        origin="lower",
        extent=_extent(grid),
        vmin=DBZ_MIN,
        vmax=DBZ_MAX,
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("East of radar (km)")
    ax.set_ylabel("North of radar (km)")
    return image


def _extent(grid: ReflectivityGrid) -> list[float]:
    half_pixel = grid.resolution_km / 2
    return [
        float(grid.x_km[0] - half_pixel),
        float(grid.x_km[-1] + half_pixel),
        float(grid.y_km[0] - half_pixel),
        float(grid.y_km[-1] + half_pixel),
    ]


def _sampled(dbz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(valid, dtype=bool) & np.isfinite(dbz), dbz, np.nan)


def plot_gap_neighbors(
    target: ReflectivityGrid,
    hidden: np.ndarray,
    prediction: np.ndarray,
    below_dbz: np.ndarray,
    below_valid: np.ndarray,
    below_elevation: float | None,
    above_dbz: np.ndarray,
    above_valid: np.ndarray,
    above_elevation: float | None,
    *,
    output_path: Path,
    unet_mae: float,
    nearest_mae: float,
    neighbor_mae: float,
    temporal_mae: float,
    precip_mae: float,
    estimated_mae: float,
    uncertain_mae: float,
    caption: str | None = None,
) -> plt.Figure:
    """Save blockage holes filled with the help of the tilts beside the target."""
    cmap = reflectivity_cmap().copy()
    cmap.set_bad("#f0f0f0")
    blocked = np.where(target.valid & ~hidden, target.dbz, np.nan)
    filled = np.where(
        target.valid & ~hidden,
        target.dbz,
        np.where(target.valid, prediction, np.nan),
    )
    error = np.where(hidden & target.valid, np.abs(prediction - target.dbz), np.nan)
    below_title = "No tilt below" if below_elevation is None else f"Tilt below {below_elevation:.2f}°"
    above_title = "No tilt above" if above_elevation is None else f"Tilt above {above_elevation:.2f}°"

    fig, axes = plt.subplots(2, 3, figsize=(14, 9.2), constrained_layout=True)
    _reflectivity(axes[0, 0], target, _sampled(below_dbz, below_valid), below_title, cmap)
    _reflectivity(
        axes[0, 1],
        target,
        blocked,
        f"Blockages in {target.sweep_elevation:.2f}°",
        cmap,
    )
    image = _reflectivity(axes[0, 2], target, _sampled(above_dbz, above_valid), above_title, cmap)
    fig.colorbar(image, ax=axes[0, 2], fraction=0.046, pad=0.04, label="dBZ")
    _reflectivity(axes[1, 0], target, filled, "Observed + U-Net fill", cmap)

    finite_error = error[np.isfinite(error)]
    error_vmax = 20.0
    if finite_error.size:
        error_vmax = float(np.clip(np.percentile(finite_error, 98), 5.0, 40.0))
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("#f0f0f0")
    error_image = axes[1, 1].imshow(
        np.ma.masked_invalid(error),
        origin="lower",
        extent=_extent(target),
        vmin=0,
        vmax=error_vmax,
        cmap=error_cmap,
        interpolation="nearest",
    )
    axes[1, 1].set_title("Absolute error in the gaps")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_xlabel("East of radar (km)")
    axes[1, 1].set_ylabel("North of radar (km)")
    fig.colorbar(error_image, ax=axes[1, 1], fraction=0.046, pad=0.04, label="dBZ")

    scores = [
        ("Nearest observed", nearest_mae, "#9e9ac8"),
        ("Neighbor average", neighbor_mae, "#b279a2"),
        ("Temporal mean", temporal_mae, "#72b7b2"),
        ("U-Net, gaps", unet_mae, "#54a24b"),
        ("U-Net, >= 20 dBZ", precip_mae, "#4c78a8"),
        ("U-Net, estimated", estimated_mae, "#F58518"),
        ("U-Net, uncertain", uncertain_mae, "#E45756"),
    ]
    names = [name for name, value, _color in scores if np.isfinite(value)]
    values = [value for _name, value, _color in scores if np.isfinite(value)]
    colors = [color for _name, value, color in scores if np.isfinite(value)]
    axes[1, 2].barh(names[::-1], values[::-1], color=colors[::-1])
    axes[1, 2].set_xlabel("MAE (dBZ)")
    axes[1, 2].set_title("Gap error")
    if values:
        axes[1, 2].set_xlim(0, max(values) * 1.45)
    for index, value in enumerate(values[::-1]):
        axes[1, 2].text(value, index, f" {value:.1f}", va="center", fontsize=8)

    fig.suptitle(
        f"{target.radar_name}  {target.timestamp}  "
        f"gaps in {target.sweep_elevation:.2f}°"
    )
    if caption:
        fig.supxlabel(caption, fontsize=9)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    return fig
