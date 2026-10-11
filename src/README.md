# Código fuente

Contiene el código recibido para el motor de contexto y el estudio de
sensibilidad H3:

- `codepysensibilidad.py`: motor, codificación semántica y telemetría.
- `sensibilidad_h3_paralelo.py`: ejecución de la malla factorial 5x5.
- `campana_replica.py`: ejecuta los 80 turnos en las dos condiciones (línea
  base y sistema con motor) y registra la telemetría por turno.
- `analizar_h2.py`: contraste de latencia y emisiones entre condiciones.
- `exportar_ratings.py`: exporta las calificaciones de los jueces a
  `data/evaluation/ratings.csv` y reporta la concordancia entre evaluadores.


