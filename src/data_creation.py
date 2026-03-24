import pandas as pd
import numpy as np


def generate_energy_data(n_days: int = 365 * 3, seed: int = 42) -> pd.DataFrame:
    """
    Génère une série temporelle réaliste de consommation d'énergie (kWh/jour).
    Composantes : tendance + saisonnalité annuelle + saisonnalité hebdomadaire + bruit.
    """
    np.random.seed(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")

    trend = np.linspace(200, 280, n_days)
    annual = 60 * np.sin(2 * np.pi * np.arange(n_days) / 365 + np.pi)
    weekly = -15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = np.random.normal(0, 10, n_days)

    peaks = np.zeros(n_days)
    for i in [15, 380, 745]:
        peaks[i : i + 5] += 40

    energy = np.clip(trend + annual + weekly + noise + peaks, 80, 450)

    return pd.DataFrame({"date": dates, "consommation_kwh": np.round(energy, 2)})
