"""
Exporta las calificaciones de los jueces J1, J2 y J3 desde la hoja de evaluación
a data/evaluation/ratings.csv y reporta la concordancia entre evaluadores.

Uso (desde la raíz del repositorio):
    python src/exportar_ratings.py
"""
import csv
import itertools
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import openpyxl
from sklearn.metrics import cohen_kappa_score

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(RAIZ, "data", "evaluation", "Evaluacion_Coherencia_Jueces_LLENA.xlsx")
CSV_OUT = os.path.join(RAIZ, "data", "evaluation", "ratings.csv")

HOJAS = {"Juez_1": "J1", "Juez_2": "J2", "Juez_3": "J3"}
CRITERIOS = ["pertinencia", "continuidad", "correccion", "extension", "naturalidad"]
# Columnas de la hoja: 8-12 = Respuesta 1, 13-17 = Respuesta 2, 18 = comentario.
# Respuesta 1 y Respuesta 2 corresponden a los sufijos -A y -B de responses.csv.
POSICIONES = {"A": 8, "B": 13}
COL_COMENTARIO = 18


def leer_calificaciones():
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    filas = []
    for hoja, juez in HOJAS.items():
        for r in wb[hoja].iter_rows(min_row=2, values_only=True):
            if not r[0]:
                continue
            for sufijo, inicio in POSICIONES.items():
                for k, criterio in enumerate(CRITERIOS):
                    filas.append({
                        "judge_id": juez,
                        "response_id": f"{r[0]}-{sufijo}",
                        "criterion": criterio,
                        "score": int(r[inicio + k]),
                        "comment": r[COL_COMENTARIO] or "",
                    })
    return filas


def agregada_por_respuesta(filas, juez):
    """Media de los cinco criterios por respuesta, en orden estable."""
    acum = {}
    for f in filas:
        if f["judge_id"] == juez:
            acum.setdefault(f["response_id"], []).append(f["score"])
    return np.array([np.mean(acum[k]) for k in sorted(acum)])


def main():
    filas = leer_calificaciones()
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["judge_id", "response_id", "criterion", "score", "comment"])
        w.writeheader()
        w.writerows(filas)
    print(f"Exportadas {len(filas)} calificaciones a {os.path.relpath(CSV_OUT, RAIZ)}")

    # Kappa ponderado cuadrático sobre la puntuación agregada redondeada (160 respuestas por juez)
    agg = {j: agregada_por_respuesta(filas, j) for j in HOJAS.values()}
    kappas = []
    for a, b in itertools.combinations(HOJAS.values(), 2):
        k = cohen_kappa_score(np.rint(agg[a]).astype(int), np.rint(agg[b]).astype(int), weights="quadratic")
        kappas.append(k)
        print(f"  kappa_w {a}-{b}: {k:.2f}")
    print(f"  kappa_w promedio: {np.mean(kappas):.2f}")

    media = np.mean([agg[j] for j in agg], axis=0)
    print(f"  Puntuación agregada global: {media.mean():.2f} (n = {len(media)} respuestas)")


if __name__ == "__main__":
    main()
