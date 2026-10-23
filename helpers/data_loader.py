from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from typing import Literal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment


SPLITS = Literal['test','train','valid']


@dataclass
class Sample:
    """One image and its ground truth; box coordinates are in pixels."""

    image_id: str
    image_path: Path
    original_mode: str
    rgb: np.ndarray
    gray: np.ndarray
    labels_yolo: np.ndarray
    boxes_xyxy: np.ndarray

    def show(self, figsize=(9, 7), show_info=False):
        """Display the image with ground-truth boxes and optional metadata."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.gray, cmap="gray", vmin=0, vmax=255)

        for x1, y1, x2, y2 in self.boxes_xyxy:
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                edgecolor="lime", facecolor="none", linewidth=1.5,
            ))

        ax.set_title(
            f"{self.image_id} — {len(self.boxes_xyxy)} annotated ships"
        )
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        plt.show()

        if show_info:
            print("Original image mode:", self.original_mode)
            print("Array shape:", self.gray.shape)
            print("Data type:", self.gray.dtype)
            print("Brightness range:", self.gray.min(), self.gray.max())
            print("Boxes in pixels:\n", self.boxes_xyxy)

        return fig, ax



    def evaluate(self, predicted_boxes, iou_threshold=0.5, plot=False):
        """
        Compare predicted boxes with this sample's ground-truth boxes.

        Boxes must use pixel coordinates: [x1, y1, x2, y2].
        A match requires IoU >= iou_threshold.
        """

        predictions = np.asarray(predicted_boxes, dtype=float)

        # Check for errors
        if not 0 < iou_threshold <= 1: raise ValueError("iou_threshold must be in (0, 1].")
        if predictions.size == 0: predictions = np.empty((0, 4), dtype=float)
        if predictions.ndim != 2 or predictions.shape[1] != 4: raise ValueError("predicted_boxes must have shape (N, 4).")
        if not np.isfinite(predictions).all(): raise ValueError("Predicted coordinates must be finite.")
        if np.any(predictions[:, 2:] <= predictions[:, :2]): raise ValueError("Each box must have x2 > x1 and y2 > y1.")



        truth = np.asarray(self.boxes_xyxy, dtype=float)
        n_predictions = len(predictions)
        n_truth = len(truth)

        # Pairwise IoU matrix: rows are predictions, columns are labels.
        top_left = np.maximum(predictions[:, None, :2], truth[None, :, :2],)
        bottom_right = np.minimum(predictions[:, None, 2:], truth[None, :, 2:],)

        intersection_size = np.maximum(bottom_right - top_left, 0)
        intersection = intersection_size.prod(axis=2)

        prediction_area = (predictions[:, 2:] - predictions[:, :2]).prod(axis=1)
        truth_area = (truth[:, 2:] - truth[:, :2]).prod(axis=1)
        union = (prediction_area[:, None] + truth_area[None, :] - intersection)

        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0,)

        matches = []

        if n_predictions and n_truth:
            eligible = iou >= iou_threshold

            # Prefer the greatest number of valid one-to-one matches, then prefer higher total IoU.
            bonus = min(n_predictions, n_truth) + 1
            reward = eligible * (bonus + iou)
            pred_indices, truth_indices = linear_sum_assignment(-reward)
            matches = [
                {
                    "prediction_index": int(p),
                    "label_index": int(t),
                    "iou": float(iou[p, t]),
                }
                for p, t in zip(pred_indices, truth_indices)
                if eligible[p, t]
            ]

        matched_predictions = {match["prediction_index"] for match in matches}
        matched_labels = {match["label_index"] for match in matches}

        false_positive_indices = sorted(set(range(n_predictions)) - matched_predictions)
        false_negative_indices = sorted(set(range(n_truth)) - matched_labels)

        tp = len(matches)
        fp = len(false_positive_indices)
        fn = len(false_negative_indices)

        # Use zero when a metric's denominator is zero.
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

        results = {
            "id": self.image_id,
            "path": self.image_path,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matches": matches,
            "false_positive_indices": false_positive_indices,
            "false_negative_indices": false_negative_indices,
            "iou_matrix": iou,
        }

        if plot:
            print(f'Evaluation Results:')
            [print(f'   - {key}: {value}') for key, value in results.items()]
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.imshow(self.gray, cmap="gray", vmin=0, vmax=255)

            for boxes, color, linestyle in ((truth, "lime", ":"),(predictions, "red", "-"),):
                for x1, y1, x2, y2 in boxes:
                    ax.add_patch(Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        edgecolor=color,
                        facecolor="none",
                        linestyle=linestyle,
                        linewidth=1.8,
                    ))

            ax.legend(handles=[
                Line2D([0], [0], color="lime", linestyle=":",label="Ground truth",),
                Line2D([0], [0], color="red", linestyle="-",label="Prediction",),
            ])

            ax.set_title(f"{self.image_id} — TP: {tp}, FP: {fp}, FN: {fn}")
            ax.set_xlabel("x (pixels)")
            ax.set_ylabel("y (pixels)")
            plt.show()

        return results












class DataLoader:
    """Read a dataset root containing images/<split> and labels/<split>."""

    def __init__(self, Root, show_tree=True, show_inventory=True):
        self.data_root = Path(Root)
        self.inventory = None
        if show_tree:
            print("New Data Loader Tree: \n")
            self.print_dir_tree()

        self.build_inventory(show_inventory)
        
        self.mode: SPLITS = 'train'






    def print_dir_tree(self, directory=None, prefix=""):
        """Print directories and one placeholder per file extension."""
        directory = self.data_root if directory is None else Path(directory)
        entries = list(directory.iterdir())
        dirs = sorted(d for d in entries if d.is_dir())
        extensions = sorted({
            f.suffix.lower() if f.suffix else "[no extension]"
            for f in entries if f.is_file()
        })
        items = dirs + extensions

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            if isinstance(item, Path):
                print(f"{prefix}{connector}{item.name}")
                new_prefix = prefix + ("    " if is_last else "│   ")
                self.print_dir_tree(item, new_prefix)
            else:
                print(f"{prefix}{connector}... {item}")





    def get_images(self, ext: str = ".jpg", split = None):
        """Return sorted image paths as strings without loading image data."""
        if not split: split = self.mode
        folder_path = self.data_root / "images" / split
        return [str(path) for path in sorted(folder_path.glob("*" + ext)) if path.is_file()]




    def load_sample(self, image_path_in: str | Path, label_ext: str = ".txt", split = None) -> Sample:
        """Load one image and validate its single-class YOLO annotations."""

        if not split: split = self.mode

        image_path = Path(image_path_in)
        label_path = (
            self.data_root / "labels" / split
            / f"{image_path.stem}{label_ext}"
        )
        # Missing labels are ingestion errors; empty files mean no ships.
        text = label_path.read_text().strip()
        try:
            labels = (
                np.array(
                    [list(map(float, line.split()))
                     for line in text.splitlines() if line.strip()],
                    dtype=np.float32,
                )
                if text else np.empty((0, 5), dtype=np.float32)
            )
        except ValueError as exc:
            raise ValueError(f"Malformed YOLO labels: {label_path}") from exc

        if labels.ndim != 2 or labels.shape[1] != 5:
            raise ValueError(f"Expected five columns per label: {label_path}")
        if not np.isfinite(labels).all():
            raise ValueError(f"Non-finite label values: {label_path}")
        if not np.all(labels[:, 0] == 0):
            raise ValueError(f"Expected class 0 (ship): {label_path}")
        if not np.all((labels[:, 1:] >= 0) & (labels[:, 1:] <= 1)):
            raise ValueError(f"Coordinates must be normalized to [0, 1]: {label_path}")
        if not np.all(labels[:, 3:] > 0):
            raise ValueError(f"Box sizes must be positive: {label_path}")

        with Image.open(image_path) as image:
            original_mode = image.mode
            rgb = np.array(image.convert("RGB"))
            gray = np.array(image.convert("L"))

        height, width = gray.shape
        centers = labels[:, 1:3] * [width, height]
        sizes = labels[:, 3:5] * [width, height]
        boxes_xyxy = np.column_stack([
            centers - sizes / 2,
            centers + sizes / 2,
        ])

        return Sample(
            image_id=image_path.stem,
            image_path=image_path,
            original_mode=original_mode,
            rgb=rgb,
            gray=gray,
            labels_yolo=labels,
            boxes_xyxy=boxes_xyxy,
        )




    def build_inventory(self, show_inventory, splits=("train", "valid", "test"), ext=".jpg"):
        """Validate samples, retain metadata, and print split-level counts."""
        records = []
        for split in splits:
            paths = self.get_images(split=split, ext=ext)
            if not paths:
                raise FileNotFoundError(
                    f"No {ext} images found in "
                    f"{self.data_root / 'images' / split}"
                )

            for path in paths:
                sample = self.load_sample(path, split=split)
                height, width = sample.gray.shape
                records.append({
                    "image_id": sample.image_id,
                    "split": split,
                    "image_path": str(sample.image_path),
                    "original_mode": sample.original_mode,
                    "width": width,
                    "height": height,
                    "n_ships": len(sample.boxes_xyxy),
                })

        if not records:
            raise ValueError("Provide at least one split to inventory.")

        inventory = pd.DataFrame(records)
        summary = inventory.groupby("split").agg(
            images=("image_id", "size"),
            ships=("n_ships", "sum"),
            min_width=("width", "min"),
            max_width=("width", "max"),
            min_height=("height", "min"),
            max_height=("height", "max"),
        )

        if show_inventory: 
            print('\n\nThe Complete Inventory of this DataLoader: \n')
            print(summary.to_string())
        self.inventory = inventory

