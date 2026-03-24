import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from statsmodels.tsa.stattools import adfuller, acf, pacf
import numpy as np

COLORS = {
    "data": "#4A90D9",
    "arima": "#E05C5C",
    "sarima": "#F5A623",
    "prophet": "#7ED321",
    "lstm": "#9B59B6",
    "test": "#95A5A6",
    "grid": "#EEEEEE",
    "bg": "#FAFAFA",
}
plt.rcParams.update(
    {
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.7,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def run_eda(df: pd.DataFrame, save_path: str = "output/01_eda.png"):
    fig = plt.figure(figsize=(18, 12), facecolor=COLORS["bg"])
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    col = "consommation_kwh"
    ts = df.set_index("date")[col]

    # — Série complète —
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(ts, color=COLORS["data"], linewidth=0.9, alpha=0.85)
    ax0.set_title(
        "Consommation journalière d'énergie (kWh)", fontsize=13, fontweight="bold"
    )
    ax0.set_ylabel("kWh")

    # — Distribution —
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(ts, bins=40, color=COLORS["data"], edgecolor="white", alpha=0.8)
    ax1.set_title("Distribution", fontsize=11)
    ax1.set_xlabel("kWh")

    # — Boxplot mensuel —
    ax2 = fig.add_subplot(gs[1, 1])
    df_m = df.copy()
    df_m["mois"] = df_m["date"].dt.strftime("%Y-%m")
    monthly = [g[col].values for _, g in df_m.groupby("mois")]
    ax2.boxplot(
        monthly,
        patch_artist=True,
        boxprops=dict(facecolor=COLORS["data"], alpha=0.6),
        medianprops=dict(color="navy", linewidth=2),
    )
    ax2.set_xticks(range(1, len(monthly) + 1, 3))
    ax2.set_xticklabels(
        [list(df_m["mois"].unique())[i] for i in range(0, len(monthly), 3)],
        rotation=45,
        fontsize=7,
    )
    ax2.set_title("Variabilité mensuelle", fontsize=11)

    # — ACF —
    ax3 = fig.add_subplot(gs[2, 0])
    acf_vals = acf(ts, nlags=60)
    ax3.bar(range(len(acf_vals)), acf_vals, color=COLORS["data"], alpha=0.7)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.axhline(1.96 / np.sqrt(len(ts)), color="red", linestyle="--", linewidth=0.8)
    ax3.axhline(-1.96 / np.sqrt(len(ts)), color="red", linestyle="--", linewidth=0.8)
    ax3.set_title("ACF (autocorrélation)", fontsize=11)
    ax3.set_xlabel("Lag")

    # — PACF —
    ax4 = fig.add_subplot(gs[2, 1])
    pacf_vals = pacf(ts, nlags=40)
    ax4.bar(range(len(pacf_vals)), pacf_vals, color=COLORS["sarima"], alpha=0.7)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.axhline(1.96 / np.sqrt(len(ts)), color="red", linestyle="--", linewidth=0.8)
    ax4.axhline(-1.96 / np.sqrt(len(ts)), color="red", linestyle="--", linewidth=0.8)
    ax4.set_title("PACF (autocorrélation partielle)", fontsize=11)
    ax4.set_xlabel("Lag")

    plt.suptitle(
        "Analyse Exploratoire — Consommation Énergie",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Test ADF de stationnarité
    result = adfuller(ts)
    print("\n── Test de Dickey-Fuller (ADF) ──")
    print(f"   Statistique : {result[0]:.4f}")
    print(f"   p-value     : {result[1]:.4f}")
    print(
        f"   Stationnaireé : {'OUI ✓' if result[1] < 0.05 else 'NON — différenciation nécessaire'}"
    )
    return save_path


def plot_forecasts(
    train, test, preds: dict, save_path: str = "output/02_forecasts.png"
):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor=COLORS["bg"])
    axes = axes.flatten()
    zoom = 120  # jours de contexte train visibles

    for ax, (name, pred) in zip(axes, preds.items()):
        color = COLORS[name.lower()]
        ax.plot(
            train.iloc[-zoom:],
            color=COLORS["data"],
            linewidth=1.2,
            alpha=0.6,
            label="Train (contexte)",
        )
        ax.plot(test, color=COLORS["test"], linewidth=1.5, label="Réel (test)")
        ax.plot(
            pred, color=color, linewidth=2, linestyle="--", label=f"Prévision {name}"
        )
        ax.axvline(test.index[0], color="black", linestyle=":", linewidth=0.8)
        ax.set_title(name, fontsize=12, fontweight="bold", color=color)
        ax.legend(fontsize=8)
        ax.set_ylabel("kWh")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(
        "Comparaison des Prévisions — 90 jours", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_metrics(results_df: pd.DataFrame, save_path: str = "output/03_metrics.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=COLORS["bg"])
    model_colors = [COLORS[m.lower()] for m in results_df["Modèle"]]

    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE (%)"]):
        bars = ax.bar(
            results_df["Modèle"],
            results_df[metric],
            color=model_colors,
            edgecolor="white",
            linewidth=0.8,
        )
        best = results_df[metric].idxmin()
        bars[best].set_edgecolor("gold")
        bars[best].set_linewidth(3)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle(
        "Comparaison des Métriques (⭐ = meilleur modèle)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_residuals(test, preds: dict, save_path: str = "output/04_residuals.png"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), facecolor=COLORS["bg"])
    axes = axes.flatten()

    for ax, (name, pred) in zip(axes, preds.items()):
        residuals = test - pred
        color = COLORS[name.lower()]
        ax.plot(residuals, color=color, linewidth=1, alpha=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.fill_between(
            residuals.index, residuals, 0, where=(residuals > 0), alpha=0.2, color=color
        )
        ax.fill_between(
            residuals.index, residuals, 0, where=(residuals < 0), alpha=0.2, color="red"
        )
        ax.set_title(f"Résidus — {name}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Erreur (kWh)")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Analyse des Résidus", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
