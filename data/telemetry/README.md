# Telemetría por turno

`turn_metrics.csv` sigue siendo una plantilla. Cada fila debe corresponder a un
turno publicado en `data/experimental/turns.csv`. Las unidades esperadas son
milisegundos para latencia, gramos de CO2 equivalente para emisiones y valores
sin unidad para entropía y fidelidad.

La tesis reporta resultados agregados de CodeCarbon, pero la exportación cruda
por turno todavía no está incluida. CodeCarbon estima consumo a partir de CPU y
RAM; no es una medición eléctrica directa.

`sensitivity_grid.csv` contiene la matriz 5x5 exportada del estudio de
sensibilidad H3 sobre 23 transiciones por configuración. La figura asociada
está en `figures/`.
