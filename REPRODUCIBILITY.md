# Reproducibilidad

## Archivos

1. `data/experimental/queries.csv` y `turns.csv`: 57 consultas (diálogos) y 80
   turnos, con categoría e identificador de diálogo.
2. `data/experimental/responses.csv`: las 160 respuestas evaluadas por los
   jueces. Los sufijos `-A` y `-B` corresponden a «Respuesta 1» y «Respuesta 2»
   de la hoja de evaluación ciega.
3. `data/evaluation/Evaluacion_Coherencia_Jueces_LLENA.xlsx` y `ratings.csv`:
   calificaciones de los tres jueces (J1, J2, J3) por respuesta y criterio.
4. `data/telemetry/turn_metrics.csv`: 160 registros por turno de latencia,
   emisiones, entropía y fidelidad; sus respuestas están en
   `data/experimental/responses_replica.csv`.
5. `data/telemetry/sensitivity_grid.csv` y `figures/`: malla 5x5 del estudio
   de sensibilidad H3.
6. `src/`: código fuente del sistema y de los análisis.

## Pasos

Instalar las dependencias de `requirements.txt` y, desde la raíz del
repositorio:

| Qué reproduce | Comando |
| --- | --- |
| Estudio de sensibilidad H3 (malla y figura) | `python src/sensibilidad_h3_paralelo.py` |
| Concordancia entre jueces y exportación de calificaciones | `python src/exportar_ratings.py` |
| Contraste de latencia y emisiones (H2) | `python src/analizar_h2.py` |
| Campaña comparativa (requiere Ollama con `qwen2.5:1.5b`) | `python src/campana_replica.py` |

## Notas

- La generación no fija semilla: repetir la campaña produce respuestas y cifras
  distintas. `turn_metrics.csv` corresponde a una reejecución del 10 de octubre
  de 2026 con el mismo corpus y código; sus cifras difieren de las del
  documento y llevan a la misma conclusión sobre H2.
- La latencia y las emisiones son estimaciones de CodeCarbon dependientes del
  hardware, no mediciones eléctricas directas. La columna
  `codecarbon_energy_kwh` permite recalcular las emisiones con el factor de la
  red ecuatoriana (0,1616 kgCO2eq/kWh).
- La clave que asigna «Respuesta 1» y «Respuesta 2» de cada par a la línea base
  o al sistema con motor no está incluida en este repositorio.
- El código fuente contiene 202 ejemplos de entrenamiento del clasificador; la
  validación cruzada del documento se reportó sobre 201.
