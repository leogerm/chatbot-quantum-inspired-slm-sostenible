# Telemetría

## `turn_metrics.csv`

Una fila por generación: 80 turnos x 2 condiciones = 160 filas, obtenidas con
`python src/campana_replica.py` en una reejecución de la campaña (10 de octubre
de 2026) sobre el corpus de `data/experimental/turns.csv`.

- `condition`: `baseline` (sin directiva de contexto) o `motor` (con directiva).
- `latency_ms`: latencia total del turno, en milisegundos.
- `emissions_co2e`: gramos de CO2 equivalente del turno, estimados con
  CodeCarbon.
- `codecarbon_energy_kwh`: energía del turno, para recalcular las emisiones con
  el factor de la red ecuatoriana (0,1616 kgCO2eq/kWh).
- `entropy`, `fidelity`, `dominant_dim`, `directive_label`, `break_detected`:
  métricas del motor de contexto.

`python src/analizar_h2.py` calcula los descriptivos y contrastes de latencia y
emisiones. Una generación (`Par-041`, `baseline`) tardó 108,8 s; se conserva en
el archivo.

## `sensitivity_grid.csv`

Malla 5x5 del estudio de sensibilidad H3 sobre 23 transiciones por
configuración, generada con `python src/sensibilidad_h3_paralelo.py`. La figura
asociada está en `figures/`.
