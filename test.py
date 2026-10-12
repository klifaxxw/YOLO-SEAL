import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# Directory containing this script (ultralytics-yolo11-main)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    """Return the weight file size in MB."""
    return Path(path).stat().st_size / 1024 / 1024


def get_class_name(names, idx):
    """Support class names stored as either a dict or a list."""
    if isinstance(names, dict):
        return str(names.get(idx, idx))
    if isinstance(names, (list, tuple)) and idx < len(names):
        return str(names[idx])
    return str(idx)


def safe_float(value):
    """Convert a NumPy value to a JSON-serializable Python float."""
    value = float(value)
    return value if np.isfinite(value) else None


def get_map75(box_metrics, idx=None):
    """
    Return AP at IoU=0.75.

    Columns in all_ap typically correspond to IoU thresholds
    0.50, 0.55, ..., 0.95.
    """
    all_ap = np.asarray(box_metrics.all_ap)

    if all_ap.ndim != 2 or all_ap.shape[1] <= 5:
        return float("nan")

    if idx is None:
        values = all_ap[:, 5]
    else:
        values = all_ap[idx, 5]

    if np.size(values) == 0:
        return float("nan")

    return float(np.mean(values))


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO_SEAL evaluation and metric summary"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to the trained best.pt; defaults to the weights for --exp-name",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(
            (SCRIPT_DIR.parent / "Full_dataset_2025" / "data.yaml").resolve()
        ),
        help="Path to the dataset data.yaml file",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="YOLO_SEAL",
        help="Experiment name used for the default weights path and output directory",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--project",
        type=str,
        default=str(SCRIPT_DIR / "runs" / "test"),
        help="Root directory for saving evaluation results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 half-precision inference",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save predictions in COCO JSON format (applicability depends on the dataset configuration)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not save evaluation plots",
    )

    return parser.parse_args()