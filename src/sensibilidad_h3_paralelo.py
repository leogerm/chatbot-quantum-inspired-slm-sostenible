import os
import sys

# Suprimir logs de dependencias secundarias en multiprocessing
os.environ["CODECARBON_LOG_LEVEL"] = "error"

# Ajuste de codificación en consola
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import time
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from codepysensibilidad import ContextEngine, SemanticEncoder, IntentClassifier

# ══════════════════════════════════════════════════════════════════════════════
#  1. BANCO DE DATOS DE 17 DIÁLOGOS MULTITURNO (40 TURNOS)
# ══════════════════════════════════════════════════════════════════════════════
DIALOGOS_TESIS = {
    "D01": {
        "categoria": "Tecnología",
        "turnos": [
            "Quiero aprender a programar en Python",
            "¿Por dónde debería empezar?",
            "¿Y qué editor de código me recomiendas?"
        ]
    },
    "D02": {
        "categoria": "Tecnología",
        "turnos": [
            "Necesito crear una base de datos para mi proyecto",
            "¿Qué motor conviene para pocos usuarios?"
        ]
    },
    "D03": {
        "categoria": "Tecnología",
        "turnos": [
            "Explícame qué es un bucle for",
            "¿Y cuándo conviene usar while en su lugar?",
            "Ahora cambiemos de tema, ¿cómo estuvo tu día?"
        ]
    },
    "D04": {
        "categoria": "Trabajo",
        "turnos": [
            "Tengo que preparar una presentación para el equipo",
            "¿Cuántas diapositivas serían razonables?",
            "¿Y cómo cierro la presentación?"
        ]
    },
    "D05": {
        "categoria": "Trabajo",
        "turnos": [
            "Quiero pedir un aumento de sueldo",
            "¿Cuál sería el mejor momento para plantearlo?"
        ]
    },
    "D06": {
        "categoria": "Trabajo",
        "turnos": [
            "Estoy planificando el trabajo del próximo mes",
            "Olvida eso, mejor recomiéndame una película"
        ]
    },
    "D07": {
        "categoria": "Estudio",
        "turnos": [
            "Quiero entender qué es el aprendizaje automático",
            "¿Y en qué se diferencia del aprendizaje profundo?",
            "Dame un ejemplo cotidiano de cada uno"
        ]
    },
    "D08": {
        "categoria": "Estudio",
        "turnos": [
            "Tengo que hacer un ensayo sobre energías renovables",
            "¿Cómo estructuro la introducción?"
        ]
    },
    "D09": {
        "categoria": "Estudio",
        "turnos": [
            "Explícame el teorema de Pitágoras",
            "¿Para qué sirve en la vida real?"
        ]
    },
    "D10": {
        "categoria": "Bienestar",
        "turnos": [
            "Quiero mejorar mi rutina diaria",
            "¿Por dónde empiezo?",
            "¿Y cómo mantengo la constancia?"
        ]
    },
    "D11": {
        "categoria": "Bienestar",
        "turnos": [
            "Me siento abrumado con todo lo que tengo pendiente",
            "¿Cómo desconecto al terminar el día?"
        ]
    },
    "D12": {
        "categoria": "Bienestar",
        "turnos": [
            "Quiero comer de forma más equilibrada",
            "Cambiando de tema, ¿qué es un algoritmo?"
        ]
    },
    "D13": {
        "categoria": "Conversacional",
        "turnos": [
            "Hola, ¿me puedes ayudar con algo?",
            "Es sobre un problema de mi computadora",
            "No enciende la pantalla"
        ]
    },
    "D14": {
        "categoria": "Conversacional",
        "turnos": [
            "Buenas tardes",
            "Solo quería conversar un rato"
        ]
    },
    "D15": {
        "categoria": "Conversacional",
        "turnos": [
            "¿Puedes explicarme qué haces exactamente?",
            "Entiendo, gracias"
        ]
    },
    "D16": {
        "categoria": "General",
        "turnos": [
            "Me interesa el tema de la sostenibilidad",
            "¿Qué puedo hacer desde casa?"
        ]
    },
    "D17": {
        "categoria": "General",
        "turnos": [
            "Cuéntame sobre los volcanes del Ecuador",
            "¿Cuál es el más alto?"
        ]
    }
}

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_TELEMETRIA = os.path.join(RAIZ, "data", "telemetry")

ALPHAS = [0.20, 0.25, 0.30, 0.35, 0.40]
COUPLINGS = [0.30, 0.40, 0.50, 0.60, 0.70]


# ══════════════════════════════════════════════════════════════════════════════
#  2. TAREA DE EVALUACIÓN INDIVIDUAL POR PROCESO (WORKER)
# ══════════════════════════════════════════════════════════════════════════════
def evaluar_configuracion_worker(args: Tuple[float, float, List[List[np.ndarray]], int, int]) -> Dict[str, Any]:
    """
    Función pura ejecutada en paralelo por los workers de la CPU.
    Recibe la tupla (alpha, coupling, precomputed_dialogues, row_idx, col_idx).
    """
    alpha, coupling, precomputed_dialogues, row_idx, col_idx = args

    state_continuities = []
    input_continuities = []
    transitions_detail = []

    for d_idx, dialogue_vectors in enumerate(precomputed_dialogues):
        engine = ContextEngine(alpha=alpha, coupling=coupling)
        for turn_idx, psi_raw in enumerate(dialogue_vectors):
            res = engine.step(psi_raw)
            # Solo los turnos t >= 2 tienen transición desde un estado previo
            if turn_idx >= 1:
                sc = res["state_continuity"]
                ic = res["continuity"]
                state_continuities.append(sc)
                input_continuities.append(ic)
                transitions_detail.append({
                    "dialogue": d_idx + 1,
                    "turn": turn_idx + 1,
                    "state_continuity": sc,
                    "input_continuity": ic
                })

    sc_arr = np.array(state_continuities)
    ic_arr = np.array(input_continuities)

    return {
        "row_idx": row_idx,
        "col_idx": col_idx,
        "alpha": alpha,
        "coupling": coupling,
        "mean_state_continuity": float(np.mean(sc_arr)),
        "std_state_continuity": float(np.std(sc_arr)),
        "min_state_continuity": float(np.min(sc_arr)),
        "max_state_continuity": float(np.max(sc_arr)),
        "mean_input_continuity": float(np.mean(ic_arr)),
        "std_input_continuity": float(np.std(ic_arr)),
        "n_transitions": len(sc_arr)
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. ANÁLISIS TOPOLÓGICO DE COMPACIDAD Y CONTIGÜIDAD (H3)
# ══════════════════════════════════════════════════════════════════════════════
def analizar_topologia_meseta(grid: np.ndarray, umbral_meseta: float, peak_coord: Tuple[int, int]) -> Dict[str, Any]:
    """
    Verifica formalmente la compacidad y contigüidad de la meseta de estabilidad
    mediante un algoritmo de componentes conexas con conectividad 4-vecinos.
    """
    n_rows, n_cols = grid.shape
    mascara = grid >= umbral_meseta
    celdas_meseta = int(np.sum(mascara))
    porcentaje_meseta = (celdas_meseta / (n_rows * n_cols)) * 100.0

    # Algoritmo BFS para encontrar componentes conexas (4-conectividad)
    visitados = np.zeros_like(mascara, dtype=bool)
    componentes = []

    for r in range(n_rows):
        for c in range(n_cols):
            if mascara[r, c] and not visitados[r, c]:
                comp = []
                queue = deque([(r, c)])
                visitados[r, c] = True
                while queue:
                    curr_r, curr_c = queue.popleft()
                    comp.append((curr_r, curr_c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < n_rows and 0 <= nc < n_cols:
                            if mascara[nr, nc] and not visitados[nr, nc]:
                                visitados[nr, nc] = True
                                queue.append((nr, nc))
                componentes.append(comp)

    # Identificar la componente que contiene el pico máximo
    comp_optima = None
    for comp in componentes:
        if peak_coord in comp:
            comp_optima = comp
            break

    es_contigua = (len(componentes) == 1) and (comp_optima is not None and len(comp_optima) == celdas_meseta)
    tamano_comp_optima = len(comp_optima) if comp_optima else 0

    return {
        "mascara": mascara,
        "celdas_meseta": celdas_meseta,
        "porcentaje_meseta": porcentaje_meseta,
        "n_componentes": len(componentes),
        "es_contigua": es_contigua,
        "tamano_comp_optima": tamano_comp_optima,
        "componentes": componentes
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4. GENERACIÓN DEL MAPA DE CALOR EDITORIAL (300 DPI)
# ══════════════════════════════════════════════════════════════════════════════
def graficar_mapa_calor_editorial(
    grid_sc: np.ndarray,
    alphas: List[float],
    couplings: List[float],
    peak_coord: Tuple[int, int],
    mascara_meseta: np.ndarray,
    output_png: str = "heatmap_sensibilidad_h3.png",
    output_pdf: str = "heatmap_sensibilidad_h3.pdf"
):
    """
    Produce un mapa de calor con estándares editoriales para tesis de grado/maestría.
    """
    sns.set_theme(style="white", font="sans-serif")
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)

    # Paleta elegante continua
    cmap = sns.color_palette("crest", as_cmap=True)

    # Heatmap base con anotaciones de 4 decimales
    sns.heatmap(
        grid_sc,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        cbar_kws={
            "label": r"Continuidad Media del Estado $\bar{C}(\alpha, c) = \psi_t \cdot \psi_{t-1}$",
            "shrink": 0.85
        },
        xticklabels=[f"{c:.2f}" for c in couplings],
        yticklabels=[f"{a:.2f}" for a in alphas],
        ax=ax,
        linewidths=1.2,
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"}
    )

    # Resaltar la celda óptima con una estrella dorada/blanca
    peak_r, peak_c = peak_coord
    ax.scatter(
        peak_c + 0.5, peak_r + 0.22,
        marker="*", s=260, color="#FFD700", edgecolor="#1a1a1a",
        linewidth=1.2, zorder=10, label=f"Óptimo ({alphas[peak_r]:.2f}, {couplings[peak_c]:.2f})"
    )

    # Dibujar bordes de contorno para las celdas de la meseta
    n_r, n_c = grid_sc.shape
    for r in range(n_r):
        for c in range(n_c):
            if mascara_meseta[r, c]:
                # Rectángulo resaltado
                rect = plt.Rectangle(
                    (c, r), 1, 1,
                    fill=False, edgecolor="#002D62",
                    linewidth=2.0, linestyle="--", zorder=5
                )
                ax.add_patch(rect)

    ax.set_title(
        r"Sensibilidad factorial de la continuidad media del estado ($\psi_t \cdot \psi_{t-1}$)" + "\n" +
        r"en el plano $(\alpha, c)$ — subconjunto multiturno (23 transiciones por celda)",
        fontsize=12, pad=18, weight="bold", color="#111827"
    )
    ax.set_xlabel(r"Fuerza de Acoplamiento Semántico $c$ (Matriz $\mathbf{C}$)", fontsize=11, labelpad=10, weight="semibold")
    ax.set_ylabel(r"Factor EWMA de Memoria Conversacional $\alpha$", fontsize=11, labelpad=10, weight="semibold")

    # Leyenda personalizada discreta
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', label=r'Pico Máximo ($\star$)',
                   markerfacecolor='#FFD700', markeredgecolor='black', markersize=14),
        plt.Line2D([0], [0], color='#002D62', linestyle='--', linewidth=2,
                   label=r'Banda de Estabilidad ($\Delta \leq 0.05$)')
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.35, 1.05), frameon=True, facecolor="#F8FAFC")

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Gráfica editorial guardada en:\n       - {output_png}\n       - {output_pdf}")



def ejecutar_sensibilidad_h3():
    print("═" * 76)
    print("  ESTUDIO DE SENSIBILIDAD H3 — MALLA FACTORIAL 5 x 5 SOBRE (α, c)")
    print("═" * 76)

    n_cpus = os.cpu_count() or 4
    print(f"  * Núcleos de CPU detectados para ejecución en paralelo: {n_cpus}")
    print(f"  * Malla Factorial: {len(ALPHAS)} x {len(COUPLINGS)} = {len(ALPHAS)*len(COUPLINGS)} configuraciones")
    print(f"  * Banco de prueba: {len(DIALOGOS_TESIS)} diálogos multiturno (40 turnos en total)")

    # ── Fase 1: Pre-codificación estática (Optimización NLP) ──
    print("\n  [1/4] Pre-codificando diálogos multiturno en el espacio semántico de 12 dimensiones...")
    t0_pre = time.perf_counter()
    encoder = SemanticEncoder()
    classifier = IntentClassifier()

    precomputed_dialogues = []
    total_turns = 0
    for d_id, data in DIALOGOS_TESIS.items():
        dialogue_vecs = []
        for msg in data["turnos"]:
            total_turns += 1
            lbl, conf, probs = classifier.predict(msg)
            raw, cov = encoder.encode(msg, probs)
            dialogue_vecs.append(raw)
        precomputed_dialogues.append(dialogue_vecs)

    t_pre_ms = (time.perf_counter() - t0_pre) * 1000.0
    print(f"        Codificados {total_turns} turnos ({len(precomputed_dialogues)} diálogos) en {t_pre_ms:.1f} ms.")

    # ── Fase 2: Ejecución en paralelo con ProcessPoolExecutor ──
    print(f"\n  [2/4] Ejecutando las 25 configuraciones en paralelo ({n_cpus} workers)...")
    tasks = []
    for r_idx, alpha in enumerate(ALPHAS):
        for c_idx, coupling in enumerate(COUPLINGS):
            tasks.append((alpha, coupling, precomputed_dialogues, r_idx, c_idx))

    t0_par = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        results = list(executor.map(evaluar_configuracion_worker, tasks))
    t_par_sec = time.perf_counter() - t0_par

    # ── Fase 3: Procesamiento de matrices y estadísticas ──
    print(f"        Cómputo en paralelo finalizado en {t_par_sec*1000.0:.2f} ms ({t_par_sec:.4f} s).")
    print("\n  [3/4] Estructurando resultados y realizando test topológico...")

    grid_state_cont = np.zeros((len(ALPHAS), len(COUPLINGS)))
    grid_state_std = np.zeros((len(ALPHAS), len(COUPLINGS)))
    grid_input_cont = np.zeros((len(ALPHAS), len(COUPLINGS)))
    rows_csv = []

    for r in results:
        i, j = r["row_idx"], r["col_idx"]
        grid_state_cont[i, j] = r["mean_state_continuity"]
        grid_state_std[i, j] = r["std_state_continuity"]
        grid_input_cont[i, j] = r["mean_input_continuity"]
        rows_csv.append(r)

    # Identificar pico máximo y meseta
    peak_val = float(np.max(grid_state_cont))
    peak_coord_idx = np.unravel_index(np.argmax(grid_state_cont), grid_state_cont.shape)
    peak_coord = (int(peak_coord_idx[0]), int(peak_coord_idx[1]))
    peak_alpha = ALPHAS[peak_coord[0]]
    peak_coupling = COUPLINGS[peak_coord[1]]

    min_val = float(np.min(grid_state_cont))
    delta_total = peak_val - min_val

    umbral_meseta = peak_val - 0.05
    topologia = analizar_topologia_meseta(grid_state_cont, umbral_meseta, peak_coord)

    # Exportar CSV
    for row in rows_csv:
        row["en_meseta_estabilidad"] = row["mean_state_continuity"] >= umbral_meseta
    df_results = pd.DataFrame(rows_csv)
    df_results = df_results.sort_values(by=["alpha", "coupling"])
    csv_path = os.path.join(DIR_TELEMETRIA, "sensitivity_grid.csv")
    df_results.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"        Exportados datos a: {csv_path}")

    # ── Fase 4: Graficación editorial ──
    print("\n  [4/4] Generando visualización científica de alta resolución...")
    os.makedirs(os.path.join(DIR_TELEMETRIA, "figures"), exist_ok=True)
    png_path = os.path.join(DIR_TELEMETRIA, "figures", "heatmap_sensibilidad_h3.png")
    pdf_path = os.path.join(DIR_TELEMETRIA, "figures", "heatmap_sensibilidad_h3.pdf")
    graficar_mapa_calor_editorial(
        grid_sc=grid_state_cont,
        alphas=ALPHAS,
        couplings=COUPLINGS,
        peak_coord=peak_coord,
        mascara_meseta=topologia["mascara"],
        output_png=png_path,
        output_pdf=pdf_path
    )

    # ── Presentación de Resultados en Consola ──
    print("\n" + "═"*76)
    print("  RESULTADOS MATRICIALES (CONTINUIDAD DE ESTADO ψ_t · ψ_{t-1})")
    print("═"*76)
    cols_header = "  α \\ c   |" + "".join([f"   {c:.2f}   |" for c in COUPLINGS])
    print(cols_header)
    print("  " + "-"*72)
    for i, a in enumerate(ALPHAS):
        line = f"   {a:.2f}   |"
        for j, c in enumerate(COUPLINGS):
            val = grid_state_cont[i, j]
            star = "★" if (i, j) == peak_coord else " "
            line += f" {val:.4f}{star}|"
        print(line)
    print("═"*76)

    print("\n" + "═"*76)
    print("  VERIFICACIÓN FORMAL DE CRITERIOS DE LA HIPÓTESIS H3")
    print("═"*76)
    print("  1. Pico de Continuidad Media (ψ_t · ψ_{t-1}):")
    print(f"     • Valor Máximo (C_max): {peak_val:.4f}")
    print(f"     • Coordenadas Óptimas:  α* = {peak_alpha:.2f}, c* = {peak_coupling:.2f}")
    print(f"     • Rango Global [Min, Max]: [{min_val:.4f}, {peak_val:.4f}] (Δ = {delta_total:.4f})")

    print(f"\n  2. Criterio de Meseta de Estabilidad (Banda ± 0.05 de C_max):")
    print(f"     • Umbral Mínimo de Estabilidad: {umbral_meseta:.4f}")
    print(f"     • Celdas en la banda:           {topologia['celdas_meseta']} de 25 ({topologia['porcentaje_meseta']:.1f}%)")
    criterio_2_ok = topologia['porcentaje_meseta'] >= 20.0
    print(f"     • Condición (≥ 20%):            {'CUMPLIDA' if criterio_2_ok else 'NO CUMPLIDA'}")

    print(f"\n  3. Criterio de Compacidad y Contigüidad Topológica:")
    print(f"     • Componentes conexas detectadas: {topologia['n_componentes']}")
    print(f"     • Celdas en componente óptima:    {topologia['tamano_comp_optima']} de {topologia['celdas_meseta']}")
    criterio_3_ok = topologia['es_contigua']
    print(f"     • Región es contigua y compacta:  {'SÍ' if criterio_3_ok else 'NO'}")

    dictamen_h3 = "CUMPLE EL CRITERIO PREREGISTRADO" if (criterio_2_ok and criterio_3_ok) else "NO CUMPLE EL CRITERIO PREREGISTRADO"
    print("\n" + "═"*76)
    print(f"  ESTADO DE H3: {dictamen_h3}")
    if umbral_meseta < min_val:
        print("  Nota: la banda de ±0.05 es más ancha que la variación total de la malla")
        print(f"        (Δ = {delta_total:.4f}); el criterio resulta poco exigente por diseño.")
    print("═"*76)


if __name__ == "__main__":
    ejecutar_sensibilidad_h3()
