"""Render the CVAT XML boxes on chosen FRONT frames (preview of what CVAT should show).

Usage (from repo root):
    python student/bonus/cvat_preview.py --xml student/bonus/cvat_annotations_109636.xml \
        --images data/bonus_cvat/images_109636 --frames 67 73 91 137 \
        --out student/bonus/cvat_preview_109636.png
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    """Draw every visible track box with its track id on the requested frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    boxes: dict[int, list[tuple[str, float, float, float, float]]] = {}
    for track in ET.parse(args.xml).getroot().findall("track"):
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            boxes.setdefault(int(box.get("frame")), []).append((
                box.find("attribute").text, *(float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))))

    fig, axes = plt.subplots(1, len(args.frames), figsize=(5 * len(args.frames), 4.2))
    for ax, frame in zip(axes, args.frames):
        img = cv2.imread(str(args.images / f"frame_{frame:06d}.jpg"))
        for tid, x0, y0, x1, y1 in boxes.get(frame, []):
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 3)
            cv2.putText(img, f"id {tid}", (int(x0), int(y0) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 255, 255), 3)
        h, w = img.shape[:2]
        crop = img[int(0.30 * h):int(0.75 * h), int(0.30 * w):int(0.75 * w)]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {frame}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
