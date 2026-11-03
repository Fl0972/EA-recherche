"""Figure de synthèse d'un banc d'essai."""

import matplotlib.pyplot as plt
import numpy as np

PANELS = [
    ("n_accuracy", "N correctement estimé", False),
    ("ospa", "Distance OSPA moyenne", True),
    ("position_error", "Erreur de position médiane\n(situations à N correct)", True),
    ("intensity_error", "Erreur d'intensité médiane\n(situations à N correct)", True),
    ("position_ratio", "Erreur de position / borne de Cramér-Rao\n(médiane par source)", True),
    ("time_ms", "Temps par situation (ms)", True),
]


def plot_benchmark(rows: list[dict], sweep_key: str, crb_label: str):
    values = sorted({r["sweep_value"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    numeric = all(isinstance(v, (int, float)) for v in values)
    log_x = numeric and min(values) > 0 and max(values) / min(values) >= 20

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (key, title, log_y) in zip(axes.flat, PANELS):
        for method in methods:
            pts = [(r["sweep_value"], r[key]) for r in rows if r["method"] == method]
            pts = [(v, m) for v, m in pts if np.isfinite(m) and (m > 0 or not log_y)]
            if not pts:
                continue
            x, y = zip(*sorted(pts))
            style = ({"c": "k", "ls": "--", "marker": None} if method == crb_label
                     else {"marker": "o"})
            ax.plot(x, y, label=method, **style)
        ax.set_title(title)
        ax.set_xlabel(sweep_key)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        if key == "n_accuracy":
            ax.set_ylim(0, 1.02)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        if key == "position_ratio":
            ax.axhspan(0.67, 0.83, color="grey", alpha=0.2, label="à la borne")
        ax.grid(alpha=0.3, which="both")
    axes.flat[2].legend()
    fig.tight_layout()
    return fig
