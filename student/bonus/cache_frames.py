"""Cache per-frame detector output and labels so bonus experiments can replay tracking.

Runs the provided LiDAR detector (Part A–C) once over the configured frame range and
stores, per frame: raw detections, valid vehicle GT boxes, FRONT camera vehicle labels
(pixel centres) and the FRONT JPEG. The cache stays under ``data/`` (gitignored) because
it contains Waymo-derived data.

Usage (from repo root):
    python student/bonus/cache_frames.py --config student/config/paths.yaml \
        --out data/bonus_cache/frames.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from fusion_lab.evaluation import valid_ground_truth
from fusion_lab.scripts.run_lab import _load_paths_config
from fusion_lab.scripts.run_lab import _resolve_weights
from fusion_lab.scripts.run_lab import _setup_import_paths
from fusion_lab.workspace_loader import load_workspace


def _box(label) -> dict:
    """Return a plain-dict copy of a Waymo 3D label box."""
    b = label.box
    return {"id": label.id, "x": b.center_x, "y": b.center_y, "z": b.center_z,
            "length": b.length, "width": b.width, "height": b.height, "heading": b.heading}


def main() -> None:
    """Run detection over the frame range and pickle the per-frame cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _setup_import_paths()
    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
    from simple_waymo_open_dataset_reader import utils as waymo_utils

    from fusion_lab.lidar_pcl import pcl_from_range_image

    cfg = _load_paths_config(args.config)
    ws = load_workspace()
    bev, det_pipe = ws["bev_mapping"], ws["detection_pipeline"]
    weights = _resolve_weights(cfg)
    det_cfg = det_pipe.load_fpn_resnet_config(str(weights) if weights else None)
    model = det_pipe.create_fpn_model(det_cfg, str(weights) if weights else None)
    tfrecord = Path(cfg["waymo_dir"]) / cfg["segment"]
    frame_start, frame_end = int(cfg["frame_start"]), int(cfg["frame_end"])
    vehicle = label_pb2.Label.Type.TYPE_VEHICLE

    cache = {"segment": tfrecord.name, "lim_x": list(det_cfg.lim_x),
             "lim_y": list(det_cfg.lim_y), "lim_z": list(det_cfg.lim_z), "frames": []}
    for cnt, frame in enumerate(WaymoDataFileReader(str(tfrecord))):
        if cnt < frame_start:
            continue
        if cnt > frame_end:
            break
        if "camera" not in cache:
            calib = waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
            cache["camera"] = {"intrinsic": list(calib.intrinsic), "width": calib.width,
                               "height": calib.height,
                               "transform": list(calib.extrinsic.transform)}
        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        tensor = torch.from_numpy(bev.bev_maps_from_pcl(points, det_cfg)).unsqueeze(0).float()
        detections = det_pipe.detect_objects_from_bev(tensor, model, det_cfg)
        labels = valid_ground_truth(frame.laser_labels, det_cfg, vehicle)
        group = next((g for g in frame.camera_labels
                      if g.name == dataset_pb2.CameraName.FRONT), None)
        camera_labels = None if group is None else [
            {"id": l.id, "u": l.box.center_x, "v": l.box.center_y,
             "w": l.box.length, "h": l.box.width}
            for l in group.labels if l.type == vehicle
        ]
        image = waymo_utils.get(frame.images, dataset_pb2.CameraName.FRONT)
        cache["frames"].append({
            "frame": cnt,
            "detections": [np.asarray(d, dtype=float) for d in detections],
            "labels": [_box(l) for l in labels],
            "camera_labels": camera_labels,
            "front_jpeg": bytes(image.image),
        })
        print(f"frame {cnt}: {len(detections)} detections, {len(labels)} GT", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump(cache, f)
    print(f"cached {len(cache['frames'])} frames -> {args.out}")


if __name__ == "__main__":
    main()
