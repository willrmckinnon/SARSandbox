import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_model_comparison(deterministic_results, yolo_results):
    models = {
        "Deterministic": deterministic_results,
        "YOLO": yolo_results,
    }
    colors = ["#64748b", "#0284c7"]

    panels = [
        (
            "Ships detected — higher is better",
            {
                "True_Positives_Detected": "True positives",
            },
            False,
        ),
        (
            "Detection scores — higher is better",
            {
                "True_Positive_Percentage": "Overall recall",
                "Average_Precision": "Mean image precision",
                "Average_Recall": "Mean image recall",
                "Average_F1": "Mean image F1",
            },
            True,
        ),
        (
            "Errors — lower is better",
            {
                "False_Positive_Detection": "False positives / labeled ships",
                "False_Negative_Detection": "Missed ships / labeled ships",
            },
            True,
        ),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5),
        gridspec_kw={"width_ratios": [1, 1.5, 1.7]},
        layout="constrained",
    )

    for ax, (title, metrics, is_percentage) in zip(axes, panels):
        positions = np.arange(len(metrics))
        bar_height = 0.34
        largest_value = 0

        for i, ((model_name, results), color) in enumerate(
            zip(models.items(), colors)
        ):
            values = [float(results[key]) for key in metrics]
            largest_value = max(largest_value, max(values))

            bars = ax.barh(
                positions + (i - 0.5) * bar_height,
                values,
                height=bar_height,
                label=model_name,
                color=color,
            )

            labels = [
                f"{value:.1%}" if is_percentage else f"{value:,.0f}"
                for value in values
            ]
            ax.bar_label(bars, labels=labels, padding=4, fontsize=9)

        ax.set_yticks(positions, list(metrics.values()))
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        if is_percentage:
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))

        # Error ratios can exceed 100%, so don't clip them.
        minimum_max = 1 if ax is axes[1] else 0
        ax.set_xlim(0, max(largest_value, minimum_max, 0.01) * 1.25)

    axes[0].legend(loc="lower right")
    fig.suptitle("SAR ship detection: deterministic vs YOLO", fontsize=15)

    plt.show()
    return fig, axes