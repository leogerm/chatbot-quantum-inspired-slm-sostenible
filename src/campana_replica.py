"""
Réplica de la campaña comparativa (línea base vs. sistema con motor de contexto).

Ejecuta los 80 turnos del corpus publicado bajo dos condiciones y registra la
telemetría por turno. La única diferencia entre condiciones es la inyección de
la directiva de contexto en el prompt de sistema; modelo, parámetros de
decodificación, ventana de historial y consulta son idénticos.

Uso (desde la raíz del repositorio, con Ollama sirviendo qwen2.5:1.5b):
    python src/campana_replica.py
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import OrderedDict

os.environ["CODECARBON_LOG_LEVEL"] = "error"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from codepysensibilidad import HybridChatbot

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDICIONES = ("baseline", "motor")
SEMILLA_ORDEN = 20261010   # solo fija qué condición se ejecuta primero en cada diálogo

CAMPOS_METRICAS = [
    "dialogue_id", "turn_id", "query_id", "category", "condition", "run_order",
    "latency_ms", "emissions_co2e", "emissions_source", "energy_kwh",
    "codecarbon_energy_kwh", "cpu_power_w", "ram_power_w",
    "response_words", "source", "entropy", "fidelity", "state_continuity",
    "dominant_dim", "directive_label", "break_detected", "ml_label",
    "ml_confidence", "temperature", "alpha", "coupling", "theta_quiebre",
]
CAMPOS_RESPUESTAS = ["response_id", "dialogue_id", "query_id", "condition", "response_text"]


def cargar_dialogos():
    base = os.path.join(RAIZ, "data", "experimental")
    with open(os.path.join(base, "queries.csv"), encoding="utf-8", newline="") as fh:
        categorias = {r["query_id"]: r["category"] for r in csv.DictReader(fh)}
    dialogos = OrderedDict()
    with open(os.path.join(base, "turns.csv"), encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            r["category"] = categorias[r["query_id"]]
            dialogos.setdefault(r["dialogue_id"], []).append(r)
    return dialogos


def energia_codecarbon(bot):
    """Energía acumulada que reporta CodeCarbon (kWh), si está disponible."""
    try:
        return float(bot.tracker._total_energy.kWh)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="número de diálogos (0 = todos)")
    ap.add_argument("--outdir", default="", help="carpeta de salida alternativa (pruebas)")
    args = ap.parse_args()

    dialogos = cargar_dialogos()
    ids = list(dialogos)[: args.limit] if args.limit else list(dialogos)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        ruta_metricas = os.path.join(args.outdir, "turn_metrics.csv")
        ruta_respuestas = os.path.join(args.outdir, "responses_replica.csv")
    else:
        ruta_metricas = os.path.join(RAIZ, "data", "telemetry", "turn_metrics.csv")
        ruta_respuestas = os.path.join(RAIZ, "data", "experimental", "responses_replica.csv")

    bot = HybridChatbot()
    if not bot.ollama_available:
        sys.exit(f"Ollama no disponible: {bot.last_llm_error}")

    # Calentamiento: carga el modelo en memoria y descarta su costo de la telemetría
    bot.respond("hola")
    bot.reset()
    if bot.tracker:
        bot._last_tracker_emissions = bot.tracker.flush() or 0.0

    rng = random.Random(SEMILLA_ORDEN)
    total = sum(len(dialogos[d]) for d in ids) * len(CONDICIONES)
    hechos = 0
    t_inicio = time.perf_counter()

    with open(ruta_metricas, "w", encoding="utf-8", newline="") as fm, \
         open(ruta_respuestas, "w", encoding="utf-8", newline="") as fr:
        wm = csv.DictWriter(fm, fieldnames=CAMPOS_METRICAS)
        wr = csv.DictWriter(fr, fieldnames=CAMPOS_RESPUESTAS)
        wm.writeheader()
        wr.writeheader()

        for d_id in ids:
            orden = list(CONDICIONES)
            rng.shuffle(orden)
            for pos, condicion in enumerate(orden, start=1):
                bot.reset()
                bot.inject_directive = (condicion == "motor")
                for turno in dialogos[d_id]:
                    e0 = energia_codecarbon(bot)
                    out = bot.respond(turno["turn_text"])
                    e1 = energia_codecarbon(bot)
                    cs = out["quantum_signal"]
                    if out["source"] != "ollama":
                        sys.exit(f"Fallo de generación en {turno['query_id']} ({condicion}): {bot.last_llm_error}")
                    wm.writerow({
                        "dialogue_id": d_id,
                        "turn_id": turno["turn_id"],
                        "query_id": turno["query_id"],
                        "category": turno["category"],
                        "condition": condicion,
                        "run_order": pos,
                        "latency_ms": out["total_ms"],
                        "emissions_co2e": out["emissions_gco2"],
                        "emissions_source": out["emissions_source"],
                        "energy_kwh": out["total_energy_kwh"],
                        "codecarbon_energy_kwh": "" if e0 is None or e1 is None else f"{e1 - e0:.10f}",
                        "cpu_power_w": out["cpu_power_w"],
                        "ram_power_w": out["ram_power_w"],
                        "response_words": len(out["response"].split()),
                        "source": out["source"],
                        "entropy": cs["entropy_vn"],
                        "fidelity": cs["fidelity"],
                        "state_continuity": cs["state_continuity"],
                        "dominant_dim": cs["dominant_dim"],
                        "directive_label": cs["directive_label"],
                        "break_detected": cs["break_detected"],
                        "ml_label": cs["ml_label"],
                        "ml_confidence": cs["ml_confidence"],
                        "temperature": round(max(0.1, min(0.9, cs["annealing_temp"])), 4),
                        "alpha": cs["alpha"],
                        "coupling": cs["coupling"],
                        "theta_quiebre": cs["theta_quiebre"],
                    })
                    wr.writerow({
                        "response_id": f"{turno['query_id']}-{condicion}",
                        "dialogue_id": d_id,
                        "query_id": turno["query_id"],
                        "condition": condicion,
                        "response_text": out["response"],
                    })
                    fm.flush()
                    fr.flush()
                    hechos += 1
                    transcurrido = time.perf_counter() - t_inicio
                    print(f"[{hechos}/{total}] {turno['query_id']} {condicion:8s} "
                          f"{out['total_ms'] / 1000:5.1f}s  {out['emissions_gco2']:.5f} g  "
                          f"({transcurrido / 60:.1f} min)", flush=True)

    print(f"Campaña completada: {hechos} generaciones en {(time.perf_counter() - t_inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
