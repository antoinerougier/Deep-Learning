"""
Implémentation de l'optimiseur Adam (Adaptive Moment Estimation)
Référence : Kingma & Ba, 2014 — https://arxiv.org/abs/1412.6980
"""

import numpy as np
import matplotlib.pyplot as plt


class Adam:

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:

        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        beta1, beta2, alpha, eps = self.beta1, self.beta2, self.lr, self.eps

        self.m = beta1 * self.m + (1 - beta1) * grads
        self.v = beta2 * self.v + (1 - beta2) * grads**2

        m_hat = self.m / (1 - beta1**self.t)
        v_hat = self.v / (1 - beta2**self.t)

        params = params - alpha * m_hat / (np.sqrt(v_hat) + eps)

        return params


def f(x):
    """Fonction objectif avec plusieurs minima locaux."""
    return x**2 + np.sin(5 * x)


def grad_f(x):
    """Gradient analytique de f."""
    return 2 * x + 5 * np.cos(5 * x)


def run_demo():
    x_range = np.linspace(-3, 3, 500)

    configs = [
        {"label": "Adam", "color": "#E8593C", "opt": Adam(lr=0.1)},
        {"label": "SGD", "color": "#3B8BD4", "opt": None, "lr": 0.05},
    ]

    n_steps = 100
    x0 = 2.5

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#f8f8f6")
    for ax in axes:
        ax.set_facecolor("#f8f8f6")

    axes[0].plot(x_range, f(x_range), color="#888", linewidth=1.5, label="f(x)")
    axes[0].set_title("Trajectoires de descente", fontsize=13, pad=10)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")

    axes[1].set_title("Convergence : f(x) au fil des itérations", fontsize=13, pad=10)
    axes[1].set_xlabel("Itération")
    axes[1].set_ylabel("f(x)")

    for cfg in configs:
        x = x0
        trajectory_x = [x]
        trajectory_y = [f(x)]

        # Adam
        if cfg["label"] == "Adam":
            opt = cfg["opt"]
            for _ in range(n_steps):
                g = grad_f(np.array([x]))
                x_arr = opt.step(np.array([x]), g)
                x = float(x_arr[0])
                trajectory_x.append(x)
                trajectory_y.append(f(x))

        # SGD
        elif cfg["label"] == "SGD":
            lr = cfg["lr"]
            for _ in range(n_steps):
                x = x - lr * grad_f(x)
                trajectory_x.append(x)
                trajectory_y.append(f(x))

        axes[0].plot(
            trajectory_x,
            trajectory_y,
            color=cfg["color"],
            linewidth=1.8,
            label=cfg["label"],
            alpha=0.85,
        )
        axes[0].scatter(
            trajectory_x[0],
            trajectory_y[0],
            color=cfg["color"],
            zorder=5,
            s=60,
            marker="o",
        )

        axes[1].plot(
            trajectory_y,
            color=cfg["color"],
            linewidth=1.8,
            label=cfg["label"],
            alpha=0.85,
        )

    x_min = -0.18
    axes[0].scatter(
        [x_min],
        [f(x_min)],
        color="black",
        zorder=6,
        s=80,
        marker="*",
        label="min global (approx.)",
    )

    for ax in axes:
        ax.legend(fontsize=10, framealpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=2)
    plt.savefig("output/adam_demo.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_demo()
