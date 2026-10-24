"""Save one cividis SAR image with ground-truth, deterministic, and YOLO boxes."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from deterministic_model import mean_threshold_model
from helpers.model import Model
from helpers.data_loader import DataLoader  # Rename data_loader(1).py to data_loader.py if needed.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def save_detection_overlay(
    sample,
    model,
    output_path="graphics/model_comparison.png",
    confidence=0.25,
    scale=3,
    show=True
):
    # Run both detection models
    boxes = {
        "Ground Truth": sample.boxes_xyxy,
        "Deterministic": mean_threshold_model(sample),
        "YOLO": model.inference(
            sample.image_path,
            confidence=confidence
        )["predicted_boxes"]
    }

    # Bounding box colors
    colors = {
        "Ground Truth": "#9bff78",
        "Deterministic": "#ff6e64",
        "YOLO": "#6eebff"
    }

    # Initialize figure
    gray = np.asarray(sample.gray)
    height, width = gray.shape
    dpi = 150

    fig, ax = plt.subplots(
        figsize=(width * scale / dpi, height * scale / dpi),
        dpi=dpi
    )

    # Display SAR imagery
    ax.imshow(
        gray,
        cmap="cividis",
        vmin=0,
        vmax=255,
        interpolation="nearest"
    )

    # Plot bounding boxes
    # Ground truth is last to improve visibility
    for label in ("Deterministic", "YOLO", "Ground Truth"):
        for x1, y1, x2, y2 in boxes[label]:
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    edgecolor=colors[label],
                    facecolor="none",
                    linewidth=1.8
                )
            )

    # Standard Matplotlib legend
    handles = [
        Line2D(
            [0], [0],
            color=colors[label],
            linewidth=2,
            label=label
        )
        for label in ("Ground Truth", "Deterministic", "YOLO")
    ]

    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=9
    )

    # Remove axes, titles, and whitespace
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save PNG
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0
    )

    if show:
        plt.show()

    plt.close(fig)
    return output_path