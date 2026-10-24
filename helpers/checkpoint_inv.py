import torch
import pandas as pd
from pathlib import Path
from IPython.display import display


def make_inventory():
    records = []

    for path in sorted(Path.cwd().rglob("*.pt")):
        if path.name not in {"best.pt", "last.pt"}:
            continue

        try:
            # Load only checkpoints you created or trust.
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            args = checkpoint.get("train_args") or {}
            metrics = checkpoint.get("train_metrics") or {}
            epoch = checkpoint.get("epoch")

            records.append({
                "path": str(path),
                "map50_95": metrics.get("metrics/mAP50-95(B)"),
                "map50": metrics.get("metrics/mAP50(B)"),
                "precision": metrics.get("metrics/precision(B)"),
                "recall": metrics.get("metrics/recall(B)"),
                "saved_epoch": (
                    epoch + 1
                    if isinstance(epoch, int) and epoch >= 0
                    else epoch
                ),
                "planned_epochs": args.get("epochs"),
                "run_name": args.get("name"),
                "original_save_dir": args.get("save_dir"),
                "dataset": args.get("data"),
                "has_optimizer": checkpoint.get("optimizer") is not None,
                "best_fitness": checkpoint.get("best_fitness"),
                "status": "loaded",
            })

            del checkpoint

        except Exception as exc:
            records.append({
                "path": str(path),
                "status": f"FAILED: {exc}",
            })

    inventory = pd.DataFrame(records)

    if not inventory.empty:
        # Missing scores, including failed checkpoints, sort last.
        score_columns = ["map50_95", "map50", "precision", "recall"]

        for column in score_columns:
            if column not in inventory.columns:
                inventory[column] = float("nan")
            inventory[column] = pd.to_numeric(
                inventory[column], errors="coerce"
            )

        inventory = inventory.sort_values(
            "map50_95",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    display(inventory)
    return inventory