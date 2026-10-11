# chatbot-quantum-inspired-slm-sostenible

Chatbot híbrido quantum-inspired con Small Language Models locales para una
inteligencia artificial conversacional sostenible.

Repositorio público: https://github.com/leogerm/chatbot-quantum-inspired-slm-sostenible

## Resumen de investigación

El objetivo es diseñar, implementar y evaluar un chatbot conversacional
quantum-inspired en español, con un Small Language Model local, un motor de
contexto semántico de 12 dimensiones y un clasificador de intención clásico,
operando en hardware doméstico sin GPU ni dependencia de la nube.

El repositorio materializa cuatro objetivos:

1. Una arquitectura modular de tres capas: clasificación, contexto y
   generación local, con degradación gradual del servicio.
2. Un motor que actualiza el estado turno a turno mediante EWMA y una matriz de
   acoplamiento 12x12, y expone entropía, continuidad y dimensión dominante.
3. Una evaluación frente a una línea base sobre el mismo corpus, incluyendo
   coherencia, latencia, emisiones estimadas y sensibilidad de hiperparámetros.
4. Un panel de inspección para observar las métricas internas y la telemetría.

### Hipótesis y estado

| Hipótesis | Enunciado | Estado |
| --- | --- | --- |
| H1 | El motor mejora la coherencia conversacional frente a la línea base | No sostenida: sin diferencia detectable entre condiciones |
| H2 | Sobrecarga menor de 500 ms y de 1 gCO2eq por turno | Compatible con la evidencia, no demostrada: no se observa sobrecarga |
| H3 | Región compacta y contigua de estabilidad en `(alpha, c)` | Cumple el criterio preregistrado; criterio poco exigente por diseño |

## Material experimental

| Evidencia | Ubicación |
| --- | --- |
| Corpus: 57 consultas, 80 turnos, con categoría e identificador de diálogo | `data/experimental/queries.csv`, `turns.csv` |
| 160 respuestas evaluadas por los jueces | `data/experimental/responses.csv` |
| Calificaciones de J1, J2 y J3 por respuesta y criterio | `data/evaluation/` |
| Registros por turno de latencia, emisiones, entropía y fidelidad | `data/telemetry/turn_metrics.csv` |
| Malla 5x5 del estudio de sensibilidad H3 y su figura | `data/telemetry/sensitivity_grid.csv`, `figures/` |
| Código fuente | `src/` |

Los registros por turno provienen de una reejecución de la campaña con el mismo
corpus y código (`src/campana_replica.py`). Como la generación no fija semilla,
sus cifras difieren de las reportadas en el documento y conducen a la misma
conclusión sobre H2. Los detalles están en `REPRODUCIBILITY.md`.

No se publican secretos, modelos binarios ni documentos personales.
