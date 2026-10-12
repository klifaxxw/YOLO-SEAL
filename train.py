"""
YOLO11 Object Detection Training Script

Features:
- Configurable training parameters through command-line arguments
- Optional pretrained checkpoint loading
- Explicit validation after training
- Metrics, runtime, and environment reporting
- JSON and TXT experiment summaries

Usage:
    python train.py --data path/to/data.yaml
    python train.py --data path/to/data.yaml --epochs 1
    python train.py --data path/to/data.yaml --weights yolo11n.pt
"""

import argparse
import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a YOLO11 object detection model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ultralytics/cfg/models/11/yolo11.yaml",
        help="Model YAML configuration or model checkpoint.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional pretrained checkpoint to initialize the model.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the dataset YAML configuration.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument(
        "--cache",
        choices=["false", "ram", "disk"],
        default="disk",
        help="Dataset caching strategy.",
    )
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed used by the training framework.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow using an existing experiment directory.",
    )

    return parser.parse_args()


def resolve_path(path, project_root):
    """Resolve relative paths against the script directory."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    model_path = resolve_path(args.model, project_root)
    data_path = resolve_path(args.data, project_root)
    project_path = resolve_path(args.project, project_root)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model configuration or checkpoint not found: {model_path}"
        )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_path}"
        )

    if args.weights:
        weights_path = resolve_path(args.weights, project_root)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {weights_path}"
            )
    else:
        weights_path = None

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Record experiment start time.
    start_time = datetime.now()
    start_timestamp = time.perf_counter()

    logger.info("Starting YOLO11 training.")
    logger.info("Model: %s", model_path)
    logger.info("Dataset: %s", data_path)
    logger.info("Device: %s", args.device)

    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("CUDA is unavailable; training may run on CPU.")

    # Initialize the model.
    torch.cuda.empty_cache()
    model = YOLO(str(model_path))

    if weights_path is not None:
        model.load(str(weights_path))
        logger.info("Initialized model with weights: %s", weights_path)
    else:
        logger.info(
            "No external checkpoint specified. "
            "A YAML model is initialized without pretrained weights."
        )

    # Convert the cache argument into the format expected by Ultralytics.
    cache = False if args.cache == "false" else args.cache

    # Train the model.
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        amp=True,
        cache=cache,
        project=str(project_path),
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
    )

    # Obtain the actual experiment directory created by Ultralytics.
    save_dir = Path(model.trainer.save_dir)
    logger.info("Training outputs: %s", save_dir)

    # Evaluate the best checkpoint when available.
    best_weights = save_dir / "weights" / "best.pt"

    if best_weights.is_file():
        eval_model = YOLO(str(best_weights))
        evaluated_checkpoint = str(best_weights)
    else:
        logger.warning(
            "Best checkpoint was not found; evaluating the current model."
        )
        eval_model = model
        evaluated_checkpoint = "current model"

    metrics = eval_model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        plots=False,
    )

    end_time = datetime.now()
    duration_seconds = time.perf_counter() - start_timestamp

    # Extract detection metrics.
    box_metrics = metrics.box

    f1_score = float(box_metrics.f1.mean())
    precision = float(box_metrics.mp)
    recall = float(box_metrics.mr)
    map50 = float(box_metrics.map50)
    map75 = float(box_metrics.map75)
    map50_95 = float(box_metrics.map)

    # Ultralytics reports validation speed in milliseconds per image.
    speed = metrics.speed
    preprocess_ms = float(speed.get("preprocess", 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))

    pipeline_ms = preprocess_ms + inference_ms + postprocess_ms

    inference_fps = (
        1000.0 / inference_ms if inference_ms > 0 else 0.0
    )
    pipeline_fps = (
        1000.0 / pipeline_ms if pipeline_ms > 0 else 0.0
    )

    # Collect experiment parameters and environment information.
    summary = {
        "model": str(model_path),
        "pretrained_weights": str(weights_path) if weights_path else None,
        "evaluated_checkpoint": evaluated_checkpoint,
        "dataset": str(data_path),
        "parameters": {
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "device": args.device,
            "optimizer": args.optimizer,
            "amp": True,
            "cache": args.cache,
            "seed": args.seed,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "mAP50": map50,
            "mAP75": map75,
            "mAP50-95": map50_95,
        },
        "speed_ms_per_image": {
            "preprocess": preprocess_ms,
            "inference": inference_ms,
            "postprocess": postprocess_ms,
            "total": pipeline_ms,
        },
        "fps": {
            "inference_only": inference_fps,
            "end_to_end_estimate": pipeline_fps,
        },
        "runtime": {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_seconds,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }

    # Save machine-readable and human-readable reports.
    json_path = save_dir / "results_summary.json"
    txt_path = save_dir / "results_summary.txt"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    report = (
        "YOLO11 Training and Evaluation Summary\n"
        "======================================\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1_score:.4f}\n"
        f"mAP50: {map50:.4f}\n"
        f"mAP75: {map75:.4f}\n"
        f"mAP50-95: {map50_95:.4f}\n"
        "\n"
        "Speed (ms/image)\n"
        "----------------\n"
        f"Preprocess: {preprocess_ms:.3f}\n"
        f"Inference: {inference_ms:.3f}\n"
        f"Postprocess: {postprocess_ms:.3f}\n"
        f"Total: {pipeline_ms:.3f}\n"
        f"Inference-only FPS: {inference_fps:.2f}\n"
        f"End-to-end estimated FPS: {pipeline_fps:.2f}\n"
        "\n"
        "Runtime\n"
        "-------\n"
        f"Start: {summary['runtime']['start_time']}\n"
        f"End: {summary['runtime']['end_time']}\n"
        f"Duration (seconds): {duration_seconds:.2f}\n"
        "\n"
        "Environment\n"
        "-----------\n"
        f"Python: {platform.python_version()}\n"
        f"PyTorch: {torch.__version__}\n"
        f"Ultralytics: {ultralytics.__version__}\n"
        f"CUDA: {torch.version.cuda}\n"
        f"GPU: {summary['environment']['gpu']}\n"
    )

    with txt_path.open("w", encoding="utf-8") as file:
        file.write(report)

    logger.info("Evaluation completed.")
    logger.info("Summary saved to: %s", json_path)
    print(report)


if __name__ == "__main__":
    main()