"""
Contraste de H2 (latencia y emisiones) sobre data/telemetry/turn_metrics.csv.

Reporta descriptivos por condición, contrastes pareados (t y Wilcoxon), intervalo
de confianza al 95 %, tamaño del efecto d_z y el desglose por categoría.

Uso (desde la raíz del repositorio):
    python src/analizar_h2.py
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_IN = os.path.join(RAIZ, "data", "telemetry", "turn_metrics.csv")
FACTOR_EMISION_EC = 161.6   # gCO2eq/kWh, red eléctrica ecuatoriana


def contraste(base: pd.Series, motor: pd.Series, nombre: str, unidad: str, dec: int):
    d = motor - base
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t = stats.ttest_rel(motor, base)
    w = stats.wilcoxon(motor, base)
    print(f"\n{nombre}")
    print(f"  baseline: media {base.mean():.{dec}f} {unidad} (DE {base.std(ddof=1):.{dec}f})")
    print(f"  motor:    media {motor.mean():.{dec}f} {unidad} (DE {motor.std(ddof=1):.{dec}f})")
    print(f"  diferencia (motor - baseline): {d.mean():+.{dec}f} {unidad} "
          f"({100 * d.mean() / base.mean():+.1f} %), DE de las diferencias {d.std(ddof=1):.{dec}f}")
    print(f"  IC 95 %: [{d.mean() - tcrit * se:+.{dec}f} ; {d.mean() + tcrit * se:+.{dec}f}] {unidad}")
    print(f"  t pareada (gl = {n - 1}): t = {t.statistic:.3f}; p = {t.pvalue:.4f}")
    print(f"  Wilcoxon: W = {w.statistic:.1f}; p = {w.pvalue:.4f}")
    print(f"  d_z = {d.mean() / d.std(ddof=1):.3f}")


def main():
    df = pd.read_csv(CSV_IN)
    df["emissions_ec"] = df["codecarbon_energy_kwh"] * FACTOR_EMISION_EC
    ancho = df.pivot(index="query_id", columns="condition")
    n = len(ancho)
    print(f"Turnos pareados: {n}; generaciones: {len(df)}; fuente de respuesta: {df['source'].value_counts().to_dict()}")
    print(f"Tiempo total de generación: {df['latency_ms'].sum() / 60000:.1f} min")
    print(f"Ruta de emisiones por turno: {df['emissions_source'].value_counts().to_dict()}")

    contraste(ancho["latency_ms"]["baseline"], ancho["latency_ms"]["motor"], "Latencia", "ms", 2)
    contraste(ancho["emissions_co2e"]["baseline"], ancho["emissions_co2e"]["motor"],
              "Emisiones (ruta del sistema)", "gCO2eq", 6)
    contraste(ancho["emissions_ec"]["baseline"], ancho["emissions_ec"]["motor"],
              "Emisiones (energía CodeCarbon x 0,1616 kgCO2eq/kWh)", "gCO2eq", 6)
    contraste(ancho["response_words"]["baseline"], ancho["response_words"]["motor"],
              "Longitud de respuesta", "palabras", 1)

    print("\nTotales de la campaña (gCO2eq):")
    for col, etiqueta in (("emissions_co2e", "ruta del sistema"), ("emissions_ec", "factor 0,1616")):
        por = df.groupby("condition")[col].sum()
        print(f"  {etiqueta}: baseline {por['baseline']:.4f}; motor {por['motor']:.4f}; total {por.sum():.4f}")
    print(f"  energía CodeCarbon total: {df['codecarbon_energy_kwh'].sum():.6f} kWh")

    print("\nCorrelaciones:")
    for c in ("baseline", "motor"):
        s = df[df["condition"] == c]
        print(f"  longitud-latencia ({c}): r = {s['response_words'].corr(s['latency_ms']):.3f}")
    print(f"  latencia-emisiones (n = {len(df)}): r = {df['latency_ms'].corr(df['emissions_co2e']):.3f}")

    print("\nPor categoría (medias):")
    cat = df.pivot_table(index="category", columns="condition",
                         values=["latency_ms", "emissions_co2e", "response_words"], aggfunc="mean")
    for metrica, dec in (("latency_ms", 0), ("emissions_co2e", 6), ("response_words", 1)):
        t = cat[metrica].copy()
        t["delta"] = t["motor"] - t["baseline"]
        print(f"  {metrica}")
        print(t.round(dec).to_string().replace("\n", "\n    ").join(["    ", ""]))

    motor = df[df["condition"] == "motor"]
    print("\nMotor de contexto (80 turnos):")
    print(f"  directivas: {motor['directive_label'].value_counts().to_dict()}")
    print(f"  dimensión dominante: {motor['dominant_dim'].value_counts().to_dict()}")
    print(f"  quiebres detectados: {int(motor['break_detected'].astype(str).eq('True').sum())}")
    print(f"  hiperparámetros: alpha {sorted(df['alpha'].unique())}, c {sorted(df['coupling'].unique())}, "
          f"theta {sorted(df['theta_quiebre'].unique())}")


if __name__ == "__main__":
    main()
