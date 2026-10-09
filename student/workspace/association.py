"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from fusion_lab.workspace_loader import load_workspace_module
from fusion_lab.workspace_support import get_tracking_params

params = get_tracking_params()
kalman = load_workspace_module("kalman")  # không dùng `import kalman`


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)
    return float((gamma.T @ np.linalg.inv(S) @ gamma).item())


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    # vi: d² ~ χ²(dim_meas) nếu cặp đúng; giữ lại gating_threshold khối xác suất.
    limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
    return bool(mhd_sq < limit)


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    costs = np.asmatrix(np.full((len(track_list), len(meas_list)), np.inf))
    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            # vi: Kiểm tra FOV trước: track sau lưng camera không được chiếu.
            if not meas.sensor.in_fov(track.x):
                continue
            mhd_sq = mahalanobis_distance(track, meas)
            if chi2_gate(mhd_sq, meas.sensor):
                costs[i, j] = mhd_sq
    return costs


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    costs = np.asarray(association_matrix, dtype=float)
    if costs.size == 0 or not np.isfinite(costs).any():
        return (np.nan, np.nan, np.asmatrix(costs),
                list(unassigned_tracks), list(unassigned_meas))
    finite = np.where(np.isfinite(costs), costs, np.inf)
    row, col = np.unravel_index(np.argmin(finite), finite.shape)
    track = unassigned_tracks[row]
    meas = unassigned_meas[col]
    reduced = np.delete(np.delete(costs, row, axis=0), col, axis=1)
    remaining_tracks = [t for k, t in enumerate(unassigned_tracks) if k != row]
    remaining_meas = [m for k, m in enumerate(unassigned_meas) if k != col]
    return track, meas, np.asmatrix(reduced), remaining_tracks, remaining_meas


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)
    costs = association_cost_matrix(unassigned_tracks, unassigned_meas)
    # vi: Greedy: lấy cặp d² nhỏ nhất, update EKF, lặp tới khi hết cặp hữu hạn.
    while costs.size > 0:
        track, meas, costs, unassigned_tracks, unassigned_meas = pick_next_pair(
            costs, unassigned_tracks, unassigned_meas
        )
        if isinstance(track, float) and np.isnan(track):
            break
        filter_obj.update(track, meas)
        manager.handle_updated_track(track, sensor)
    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
