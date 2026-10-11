# Telemetría por turno

`turn_metrics.csv` sigue siendo una plantilla. Cada fila debe corresponder a un turno
publicado en `data/experimental/turns.csv`. Las unidades sugeridas son
milisegundos para latencia, gramos de CO2e para emisiones y valores sin unidad
para entropía y fidelidad; deben confirmarse contra la instrumentación original
antes de completar los datos.

`sensitivity_grid.csv` contiene la matriz 5x5 exportada del estudio de
sensibilidad H3. La figura asociada está en `figures/`.
