"""Bonus experiments: calibration sensitivity, camera-update visualisation, CVAT export.

Replays the lab tracker (student Part E–H through the platform Sensor/Filter/TrackManager)
on the cached detector output from ``cache_frames.py``, reproducing the platform camera
noise stream (``default_rng(seed)``, ``normal(0, 0.5, 2)`` per FRONT vehicle label), so the
unperturbed replay must equal ``student/artifacts/metrics_{lidar,fused}.json``.

Usage (from repo root):
    python student/bonus/bonus_analysis.py --cache data/bonus_cache/frames.pkl \
        --out student/bonus --cvat-images data/bonus_cvat/images
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from fusion_lab import tracking_params as params  # noqa: E402
from fusion_lab.evaluation import TRACK_GATE_METERS  # noqa: E402
from fusion_lab.evaluation import _partial_assignment  # noqa: E402
from fusion_lab.evaluation import tracking_counts  # noqa: E402
from fusion_lab.export_cvat import export_tracks_json  # noqa: E402
from fusion_lab.scripts.run_lab import _lidar_observations  # noqa: E402
from fusion_lab.tracking.filter import Filter  # noqa: E402
from fusion_lab.tracking.manager import TrackManager  # noqa: E402
from fusion_lab.tracking.sensors import Sensor  # noqa: E402
from fusion_lab.workspace_loader import load_workspace  # noqa: E402

CAMERA_NOISE_PX = 0.5  # platform _front_observations noise std (pixels)
GATE_CAM = chi2.ppf(params.gating_threshold, df=2)

# Calibration error levels: (name, yaw error in degrees about camera up-axis,
# lateral offset in metres along camera left-axis). Applied to the tracker's belief
# of the extrinsic only; the measurements still come from the true camera.
CALIB_LEVELS = [
    ("baseline", 0.0, 0.0),
    ("yaw 0.25°", 0.25, 0.0),
    ("yaw 0.5°", 0.5, 0.0),
    ("yaw 1°", 1.0, 0.0),
    ("yaw 2°", 2.0, 0.0),
    ("yaw 5°", 5.0, 0.0),
    ("lateral 0.2 m", 0.0, 0.2),
    ("lateral 0.5 m", 0.0, 0.5),
    ("lateral 1.0 m", 0.0, 1.0),
]


class RecordingFilter(Filter):
    """Platform Filter that also logs camera innovations before each update."""

    def __init__(self, kalman_mod: Any) -> None:
        super().__init__(kalman_mod)
        self.camera_innovations: list[dict[str, float]] = []

    def update(self, track: Any, meas: Any) -> None:
        if meas.sensor.name == "camera":
            H = meas.sensor.get_H(track.x)
            gamma = self._k.innovation(track.x, meas)
            S = self._k.innovation_covariance(track.P, meas, H)
            nis = float((gamma.T @ np.linalg.inv(S) @ gamma).item())
            self.camera_innovations.append({
                "frame": int(round(meas.t / params.dt)), "track": track.id,
                "du": float(gamma[0, 0]), "dv": float(gamma[1, 0]), "nis": nis,
            })
        super().update(track, meas)


def _labels(frame: dict) -> list[SimpleNamespace]:
    """Wrap cached GT boxes so platform ``tracking_counts`` can read them."""
    return [SimpleNamespace(id=b["id"], box=SimpleNamespace(
        center_x=b["x"], center_y=b["y"], center_z=b["z"])) for b in frame["labels"]]


def make_camera(cache: dict, cam_mod: Any, yaw_deg: float = 0.0,
                lateral_m: float = 0.0) -> Sensor:
    """Build the FRONT camera sensor, optionally with a miscalibrated extrinsic belief.

    Args:
        cache: Frame cache with the true FRONT calibration.
        cam_mod: Student ``camera_fusion`` module.
        yaw_deg: Rotation error about the camera up axis (degrees).
        lateral_m: Translation error along the camera left axis (metres).

    Returns:
        Camera sensor whose ``veh_to_sens`` encodes the believed (wrong) extrinsic.
    """
    cal = cache["camera"]
    calib = SimpleNamespace(intrinsic=cal["intrinsic"], width=cal["width"],
                            extrinsic=SimpleNamespace(transform=cal["transform"]))
    sensor = Sensor("camera", calib, cam_mod)
    if yaw_deg or lateral_m:
        a = math.radians(yaw_deg)
        delta = np.eye(4)
        delta[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
        delta[1, 3] = lateral_m
        believed = np.asarray(sensor.sens_to_veh) @ delta
        sensor.sens_to_veh = np.asmatrix(believed)
        sensor.veh_to_sens = np.asmatrix(np.linalg.inv(believed))
    return sensor


def replay(cache: dict, ws: dict, camera: Sensor | None, seed: int = 0) -> dict:
    """Run lidar-only (camera None) or fused tracking over the cached frames.

    Args:
        cache: Frame cache.
        ws: Loaded student workspace modules.
        camera: Camera sensor for fused mode, or None for lidar-only.
        seed: Camera-noise seed (platform default 0).

    Returns:
        Dict with per-frame records, track history, camera diagnostics and totals.
    """
    rng = np.random.default_rng(seed)
    cfg = SimpleNamespace(lim_x=cache["lim_x"], lim_y=cache["lim_y"])
    lidar = Sensor("lidar", None, ws["camera_fusion"])
    kf = RecordingFilter(ws["kalman"])
    manager = TrackManager(ws["track_management"])
    assoc = ws["association"]
    records, history, gate_min = [], [], []
    for fr in cache["frames"]:
        cnt = fr["frame"]
        obs = _lidar_observations(cnt, fr["detections"], lidar, cfg)
        for track in manager.track_list:
            kf.predict(track)
            track.set_t(cnt * params.dt)
        assoc.associate_and_update(manager, obs, kf, lidar)
        after_lidar = {t.id: np.asarray(t.x).ravel()[:3].copy() for t in manager.track_list}
        cam_obs = []
        if camera is not None and fr["camera_labels"] is not None:
            for lab in fr["camera_labels"]:
                centre = np.array([lab["u"], lab["v"]])
                camera.generate_measurement(cnt, centre + rng.normal(0, CAMERA_NOISE_PX, 2),
                                            cam_obs)
            # Track-centric gate check: does each visible confirmed track have any
            # camera measurement inside its chi-square gate?
            if cam_obs:
                for t in manager.track_list:
                    if t.state == "confirmed" and camera.in_fov(t.x):
                        gate_min.append(min(assoc.mahalanobis_distance(t, m) for m in cam_obs))
            assoc.associate_and_update(manager, cam_obs, kf, camera)
        labels = _labels(fr)
        records.append({"frame": cnt, **tracking_counts(manager.track_list, labels)})
        history.append({
            "frame": cnt,
            "camera_obs": [np.asarray(m.z).ravel().tolist() for m in cam_obs],
            "tracks": [{
                "id": t.id, "state": t.state, "score": float(t.score),
                "x_after_lidar": after_lidar.get(t.id, np.full(3, np.nan)).tolist(),
                "x": np.asarray(t.x).ravel()[:3].tolist(),
                "v": np.asarray(t.x).ravel()[3:].tolist(),
                "length": t.length, "width": t.width, "height": t.height, "yaw": t.yaw,
            } for t in manager.track_list],
        })
    totals = {k: sum(r[k] for r in records)
              for k in ("confirmed", "matches", "sum_sq_err", "ghosts", "misses")}
    totals["rmse"] = math.sqrt(totals["sum_sq_err"] / totals["matches"]) if totals["matches"] else None
    return {"records": records, "history": history, "totals": totals,
            "innovations": kf.camera_innovations, "gate_min": gate_min}


def quality(totals: dict, det_tp: int) -> dict:
    """Compute the rubric quality inputs for a replay."""
    precision = totals["matches"] / max(1, totals["matches"] + totals["ghosts"])
    coverage = totals["matches"] / max(1, det_tp)
    return {"precision_track": precision, "coverage": coverage}


def check_reproduces_official(lidar_run: dict, fused_run: dict, artifacts: Path) -> None:
    """Assert the unperturbed replay matches the graded per-mode metrics exactly."""
    for mode, run in (("lidar", lidar_run), ("fused", fused_run)):
        official = json.loads((artifacts / f"metrics_{mode}.json").read_text())["tracking"][mode]
        assert run["totals"]["matches"] == official["matches"], (mode, run["totals"], official)
        assert run["totals"]["ghosts"] == official["ghost_track_frames"], mode
        assert run["totals"]["misses"] == official["missed_gt_frames"], mode
        assert math.isclose(run["totals"]["sum_sq_err"], official["sum_sq_err"], rel_tol=1e-9), mode


# ---------------------------------------------------------------- matching helpers

def gt_track_pairs(history_frame: dict, labels: list) -> dict[str, int]:
    """Map GT id -> confirmed track id using the platform 2 m XY one-to-one matching."""
    confirmed = [t for t in history_frame["tracks"] if t["state"] == "confirmed"]
    if not confirmed or not labels:
        return {}
    pos = np.array([t["x"] for t in confirmed])
    gt = np.array([(b.box.center_x, b.box.center_y) for b in labels])
    dist = np.linalg.norm(pos[:, None, :2] - gt[None, :, :], axis=2)
    pairs = _partial_assignment(dist, dist <= TRACK_GATE_METERS)
    return {labels[j].id: confirmed[i]["id"] for i, j in pairs}


def id_events(cache: dict, run: dict) -> dict:
    """Find ID switches (GT re-matched to another track id) and ghost track-frames."""
    last: dict[str, int] = {}
    switches, ghosts = [], []
    for fr, hist in zip(cache["frames"], run["history"]):
        labels = _labels(fr)
        pairs = gt_track_pairs(hist, labels)
        matched = set(pairs.values())
        for gid, tid in pairs.items():
            if gid in last and last[gid] != tid:
                switches.append({"frame": fr["frame"], "gt": gid, "from": last[gid], "to": tid})
            last[gid] = tid
        for t in hist["tracks"]:
            if t["state"] == "confirmed" and t["id"] not in matched:
                ghosts.append({"frame": fr["frame"], "track": t["id"], "x": t["x"]})
    return {"id_switches": switches, "ghosts": ghosts}


# ---------------------------------------------------------------- projection helpers

def project(points_veh: np.ndarray, sensor: Sensor) -> np.ndarray:
    """Pinhole-project Nx3 vehicle points with the sensor calibration; NaN behind camera."""
    T = np.asarray(sensor.veh_to_sens)
    p = points_veh @ T[:3, :3].T + T[:3, 3]
    uv = np.full((len(p), 2), np.nan)
    ok = p[:, 0] > 1e-6
    uv[ok, 0] = sensor.c_i - sensor.f_i * p[ok, 1] / p[ok, 0]
    uv[ok, 1] = sensor.c_j - sensor.f_j * p[ok, 2] / p[ok, 0]
    return uv


def box_corners(t: dict) -> np.ndarray:
    """Return the 8 vehicle-frame corners of a track box."""
    c, s = math.cos(t["yaw"]), math.sin(t["yaw"])
    l, w, h = t["length"] / 2, t["width"] / 2, t["height"] / 2
    local = np.array([[sx * l, sy * w, sz * h] for sx in (-1, 1) for sy in (-1, 1)
                      for sz in (-1, 1)])
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rot.T + np.asarray(t["x"])


# ---------------------------------------------------------------- outputs

def calibration_study(cache, ws, det_tp, out: Path) -> list[dict]:
    """Run each calibration level and write CSV + Markdown table + plot."""
    rows = []
    for name, yaw, lat in CALIB_LEVELS:
        run = replay(cache, ws, make_camera(cache, ws["camera_fusion"], yaw, lat))
        inn = run["innovations"]
        du = np.array([i["du"] for i in inn])
        dv = np.array([i["dv"] for i in inn])
        nis = np.array([i["nis"] for i in inn])
        gmin = np.array(run["gate_min"])
        q = quality(run["totals"], det_tp)
        rows.append({
            "level": name, "yaw_deg": yaw, "lateral_m": lat,
            "camera_updates": len(inn),
            "mean_du_px": float(du.mean()) if len(du) else float("nan"),
            "mean_dv_px": float(dv.mean()) if len(dv) else float("nan"),
            "mean_nis": float(nis.mean()) if len(nis) else float("nan"),
            "median_track_min_d2": float(np.median(gmin)) if len(gmin) else float("nan"),
            "track_gate_miss_rate": float((gmin > GATE_CAM).mean()) if len(gmin) else float("nan"),
            "rmse": run["totals"]["rmse"], "matches": run["totals"]["matches"],
            "ghosts": run["totals"]["ghosts"], "misses": run["totals"]["misses"],
            **q,
        })
        print(json.dumps(rows[-1]), flush=True)
    with (out / "calibration_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Mức lệch | Cam updates | mean γu (px) | mean γv (px) | mean NIS | median min d² (track) | % track không có đo trong cổng | RMSE (m) | matches | ghosts | misses |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['level']} | {r['camera_updates']} | {r['mean_du_px']:.2f} | {r['mean_dv_px']:.2f} | "
            f"{r['mean_nis']:.2f} | {r['median_track_min_d2']:.2f} | {100 * r['track_gate_miss_rate']:.1f}% | "
            f"{r['rmse']:.4f} | {r['matches']} | {r['ghosts']} | {r['misses']} |")
    (out / "calibration_results.md").write_text("\n".join(lines) + "\n")

    yaw_rows = [r for r in rows if r["lateral_m"] == 0]
    x = [r["yaw_deg"] for r in yaw_rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(x, [r["mean_du_px"] for r in yaw_rows], "o-", label="mean γu")
    axes[0].plot(x, [r["mean_dv_px"] for r in yaw_rows], "s-", label="mean γv")
    axes[0].set(xlabel="yaw error (°)", ylabel="mean innovation of accepted updates (px)",
                title="Innovation bias")
    axes[0].legend()
    axes[1].plot(x, [100 * r["track_gate_miss_rate"] for r in yaw_rows], "o-", color="C3")
    axes[1].set(xlabel="yaw error (°)",
                ylabel="% tracks w/o camera meas. in gate",
                title=f"Gate rejections (χ²₂ 99.5% = {GATE_CAM:.2f})")
    axes[2].plot(x, [r["rmse"] for r in yaw_rows], "o-", color="C2", label="fused RMSE")
    axes[2].axhline(rows[0]["rmse"], ls="--", color="gray", label="baseline fused")
    axes[2].set(xlabel="yaw error (°)", ylabel="RMSE (m)", title="Tracking RMSE")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Camera extrinsic yaw miscalibration: innovation, gating and RMSE")
    fig.tight_layout()
    fig.savefig(out / "bonus_calibration_sweep.png", dpi=120)
    plt.close(fig)
    return rows


def track_error_series(cache, run, gt_id) -> tuple[list, list, list]:
    """Per-frame 3D error and track id of the track matched to one GT object."""
    frames, err, tids = [], [], []
    for fr, hist in zip(cache["frames"], run["history"]):
        labels = _labels(fr)
        pairs = gt_track_pairs(hist, labels)
        if gt_id not in pairs:
            continue
        gt = next(b for b in fr["labels"] if b["id"] == gt_id)
        tr = next(t for t in hist["tracks"] if t["id"] == pairs[gt_id])
        frames.append(fr["frame"])
        err.append(float(np.linalg.norm(np.asarray(tr["x"]) - [gt["x"], gt["y"], gt["z"]])))
        tids.append(tr["id"])
    return frames, err, tids


def viz_camera_effect(cache, lidar_run, fused_run, camera, out: Path) -> dict:
    """Pick the GT car with most fused camera updates; plot errors and camera corrections."""
    # Camera correction per (frame, track) in the fused run.
    corrections = []
    for hist in fused_run["history"]:
        for t in hist["tracks"]:
            a, b = np.asarray(t["x_after_lidar"]), np.asarray(t["x"])
            if np.isfinite(a).all() and np.linalg.norm(b - a) > 1e-9:
                corrections.append({"frame": hist["frame"], "track": t["id"],
                                    "dx": (b - a).tolist()})
    # GT object matched in both modes for the most frames while camera-corrected.
    counts: dict[str, int] = {}
    corrected = {(c["frame"], c["track"]) for c in corrections}
    for fr, hist in zip(cache["frames"], fused_run["history"]):
        for gid, tid in gt_track_pairs(hist, _labels(fr)).items():
            if (fr["frame"], tid) in corrected:
                counts[gid] = counts.get(gid, 0) + 1
    gt_id = max(counts, key=counts.get)
    fl, el, tl = track_error_series(cache, lidar_run, gt_id)
    ff, ef, tf = track_error_series(cache, fused_run, gt_id)
    fused_tid = max(set(tf), key=tf.count)
    corr = [c for c in corrections if c["track"] == fused_tid]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(fl, el, ".-", label=f"LiDAR-only (track {sorted(set(tl))})")
    axes[0].plot(ff, ef, ".-", label=f"LiDAR + camera (track {sorted(set(tf))})")
    axes[0].set(ylabel="3D position error vs GT (m)",
                title=f"Same GT vehicle in both modes (GT id {gt_id[:8]}…)")
    axes[0].legend()
    axes[1].bar([c["frame"] for c in corr], [c["dx"][1] for c in corr], color="C1",
                label="Δy from camera update (after − before, m)")
    axes[1].plot([c["frame"] for c in corr], [c["dx"][0] for c in corr], "k.",
                 label="Δx from camera update (m)")
    axes[1].set(xlabel="frame", ylabel="state correction (m)",
                title=f"Camera EKF update on fused track {fused_tid}: lateral/longitudinal shift")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "bonus_track_camera_effect.png", dpi=120)
    plt.close(fig)

    # Image overlay at the frame with the largest lateral correction of that track.
    best = max(corr, key=lambda c: abs(c["dx"][1]))
    idx = next(i for i, fr in enumerate(cache["frames"]) if fr["frame"] == best["frame"])
    fr, hist = cache["frames"][idx], fused_run["history"][idx]
    tr = next(t for t in hist["tracks"] if t["id"] == fused_tid)
    img = cv2.imdecode(np.frombuffer(fr["front_jpeg"], np.uint8), cv2.IMREAD_COLOR)
    before = project(np.asarray([tr["x_after_lidar"]]), camera)[0]
    after = project(np.asarray([tr["x"]]), camera)[0]
    meas = min(hist["camera_obs"], key=lambda z: np.hypot(z[0] - after[0], z[1] - after[1]))
    corners = project(box_corners(tr), camera)
    x0, y0 = np.nanmin(corners, axis=0)
    x1, y1 = np.nanmax(corners, axis=0)
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)
    cv2.drawMarker(img, tuple(int(v) for v in before), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 28, 3)
    cv2.drawMarker(img, tuple(int(v) for v in meas), (0, 255, 0), cv2.MARKER_CROSS, 28, 3)
    cv2.drawMarker(img, tuple(int(v) for v in after), (255, 128, 0), cv2.MARKER_DIAMOND, 22, 3)
    pad = 160
    cx, cy = int(after[0]), int(after[1])
    crop = img[max(0, cy - pad):cy + pad, max(0, cx - 2 * pad):cx + 2 * pad]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"FRONT camera, frame {fr['frame']}, fused track {fused_tid}")
    axes[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[1].set_title("zoom: red × = h(x) after LiDAR update, green + = camera meas.,\n"
                      "blue ◆ = h(x) after camera update, yellow = track box")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "bonus_camera_update_overlay.png", dpi=110)
    plt.close(fig)
    return {"gt_id": gt_id, "fused_track": fused_tid, "lidar_tracks": sorted(set(tl)),
            "fused_tracks": sorted(set(tf)), "overlay_frame": fr["frame"],
            "overlay_dx": best["dx"],
            "pixel_before": before.tolist(), "pixel_meas": list(meas),
            "pixel_after": after.tolist(),
            "mean_err_lidar": float(np.mean(el)), "mean_err_fused": float(np.mean(ef)),
            "frames_matched_lidar": len(fl), "frames_matched_fused": len(ff),
            "n_camera_updates_on_track": len(corr)}


def viz_per_vehicle(cache, lidar_run, fused_run, out: Path) -> list[dict]:
    """Bar chart of mean 3D error per GT vehicle, LiDAR-only vs fused (>= 20 matched frames)."""
    gt_ids = {b["id"] for fr in cache["frames"] for b in fr["labels"]}
    rows = []
    for gid in gt_ids:
        fl, el, _ = track_error_series(cache, lidar_run, gid)
        ff, ef, _ = track_error_series(cache, fused_run, gid)
        if len(fl) >= 20 and len(ff) >= 20:
            rows.append({"gt": gid, "frames": len(ff), "rmse_lidar": float(np.sqrt(np.mean(np.square(el)))),
                         "rmse_fused": float(np.sqrt(np.mean(np.square(ef))))})
    rows.sort(key=lambda r: -r["frames"])
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = np.arange(len(rows))
    ax.bar(idx - 0.2, [r["rmse_lidar"] for r in rows], 0.4, label="LiDAR-only")
    ax.bar(idx + 0.2, [r["rmse_fused"] for r in rows], 0.4, label="LiDAR + camera")
    ax.set_xticks(idx, [f"{r['gt'][:6]}\n({r['frames']} fr)" for r in rows], fontsize=8)
    ax.set(ylabel="per-vehicle 3D RMSE (m)",
           title="Camera update helps most vehicles but not all (seed 0, frames 0–198)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "bonus_per_vehicle_rmse.png", dpi=120)
    plt.close(fig)
    return rows


def viz_innovation_hist(cache, ws, out: Path) -> None:
    """Histogram of horizontal camera innovation at 0°, 1° and 2° yaw error."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for yaw in (0.0, 1.0, 2.0):
        run = replay(cache, ws, make_camera(cache, ws["camera_fusion"], yaw, 0.0))
        du = [i["du"] for i in run["innovations"]]
        ax.hist(du, bins=60, range=(-60, 60), alpha=0.5, label=f"yaw {yaw:g}° (n={len(du)})")
    ax.set(xlabel="γu = u_meas − h_u(x) (px)", ylabel="accepted camera updates",
           title="Horizontal innovation shifts with extrinsic yaw error")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "bonus_innovation_hist.png", dpi=120)
    plt.close(fig)


def cvat_export(cache, fused_run, camera, out: Path, image_dir: Path | None,
                tag: str = "") -> dict:
    """Export tracks via platform JSON hook and a CVAT-for-video 1.1 XML on FRONT frames."""
    export_tracks_json(fused_run["history"], out / f"cvat_tracks_fused{tag}.json")
    events = id_events(cache, fused_run)
    width, height = cache["camera"]["width"], cache["camera"]["height"]
    tracks: dict[int, list[str]] = {}
    for i, hist in enumerate(fused_run["history"]):
        for t in hist["tracks"]:
            if t["state"] != "confirmed" or not camera.in_fov(np.asarray(t["x"] + [0, 0, 0])):
                continue
            uv = project(box_corners(t), camera)
            if not np.isfinite(uv).all():
                continue
            x0, y0 = np.clip(uv.min(axis=0), 0, [width - 1, height - 1])
            x1, y1 = np.clip(uv.max(axis=0), 0, [width - 1, height - 1])
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            tracks.setdefault(t["id"], []).append(
                f'    <box frame="{i}" keyframe="1" outside="0" occluded="0" '
                f'xtl="{x0:.1f}" ytl="{y0:.1f}" xbr="{x1:.1f}" ybr="{y1:.1f}" z_order="0">'
                f'<attribute name="track_id">{t["id"]}</attribute></box>')
    xml = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>",
           "  <meta><task><labels><label><name>vehicle_track</name><attributes><attribute>"
           "<name>track_id</name><input_type>number</input_type><values>0;100000;1</values>"
           "</attribute></attributes></label></labels></task></meta>"]
    for n, (tid, boxes) in enumerate(sorted(tracks.items())):
        xml.append(f'  <track id="{n}" label="vehicle_track" source="manual">')
        xml.extend(boxes)
        last = int(boxes[-1].split('frame="')[1].split('"')[0])
        if last + 1 < len(fused_run["history"]):
            xml.append(boxes[-1].replace(f'frame="{last}"', f'frame="{last + 1}"')
                       .replace('outside="0"', 'outside="1"'))
        xml.append("  </track>")
    xml.append("</annotations>")
    (out / f"cvat_annotations{tag}.xml").write_text("\n".join(xml) + "\n")
    (out / f"cvat_id_events{tag}.json").write_text(json.dumps(events, indent=2))
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(cache["frames"]):
            (image_dir / f"frame_{i:06d}.jpg").write_bytes(fr["front_jpeg"])
    return {"n_id_switches": len(events["id_switches"]), "n_ghost_frames": len(events["ghosts"]),
            "first_switches": events["id_switches"][:5], "first_ghosts": events["ghosts"][:5]}


def main() -> None:
    """Run all bonus analyses and write a summary JSON next to the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("student/bonus"))
    parser.add_argument("--artifacts", type=Path, default=Path("student/artifacts"))
    parser.add_argument("--cvat-images", type=Path, default=None)
    parser.add_argument("--cvat-only", action="store_true",
                        help="Other segment: fused replay + CVAT export/ID events only")
    parser.add_argument("--tag", default="", help="Suffix for CVAT output files")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.cache.open("rb") as f:
        cache = pickle.load(f)
    ws = load_workspace()
    det_tp = json.loads((args.artifacts / "metrics.json").read_text())["detection"]["tp"]

    camera = make_camera(cache, ws["camera_fusion"])
    if args.cvat_only:
        fused_run = replay(cache, ws, camera)
        info = {"segment": cache["segment"], **fused_run["totals"],
                **cvat_export(cache, fused_run, camera, args.out, args.cvat_images, args.tag)}
        (args.out / f"cvat_summary{args.tag}.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2))
        return
    lidar_run = replay(cache, ws, None)
    fused_run = replay(cache, ws, camera)
    check_reproduces_official(lidar_run, fused_run, args.artifacts)
    print("replay reproduces official lidar/fused metrics", flush=True)

    summary = {
        "replay_matches_official": True,
        "lidar": {**lidar_run["totals"], **quality(lidar_run["totals"], det_tp),
                  "id_switches": len(id_events(cache, lidar_run)["id_switches"])},
        "fused": {**fused_run["totals"], **quality(fused_run["totals"], det_tp),
                  "id_switches": len(id_events(cache, fused_run)["id_switches"])},
        "camera_effect": viz_camera_effect(cache, lidar_run, fused_run, camera, args.out),
        "per_vehicle": viz_per_vehicle(cache, lidar_run, fused_run, args.out),
        "calibration": calibration_study(cache, ws, det_tp, args.out),
        "cvat": cvat_export(cache, fused_run, camera, args.out, args.cvat_images),
        "gate_threshold_camera": GATE_CAM,
    }
    viz_innovation_hist(cache, ws, args.out)
    (args.out / "bonus_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "calibration"}, indent=2,
                     default=float))


if __name__ == "__main__":
    main()
