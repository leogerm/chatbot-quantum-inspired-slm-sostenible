# Reproducibilidad

## Alcance

El repositorio documenta la evidencia disponible para el apartado D.3 sin
inventar resultados ni publicar artefactos no autorizados. Incluye el corpus
independiente de 57 consultas y 80 turnos, las 160 respuestas de ambas
condiciones, el manual de evaluación ciega, el código fuente y la matriz de
sensibilidad H3. Los archivos que todavía no tienen mediciones finales se
mantienen como plantillas con sus encabezados.

## Convenciones

- `dialogue_id` identifica un diálogo y debe ser estable entre archivos.
- `query_id` identifica una consulta dentro del corpus.
- `turn_id` identifica un turno dentro de un diálogo.
- `condition` distingue las dos condiciones experimentales.
- Los textos deben conservar su codificación UTF-8 y no incluir nombres,
  correos, claves, tokens ni otros datos personales.
- Las métricas deben conservar sus unidades en el nombre o la documentación
  del campo. No se deben completar valores ausentes con ceros.

## Archivos esperados

1. `data/experimental/queries.csv`: 57 consultas exportadas con categoría e
   identificador de diálogo.
2. `data/experimental/turns.csv`: 80 turnos asociados a sus diálogos.
3. `data/experimental/responses.csv`: 160 respuestas, identificadas por
   condición.
4. `data/evaluation/manual_jueces_ciegos.xlsx`: instrumento para aplicar la
   evaluación ciega a 80 pares, con cinco criterios y escala de 1 a 5.
5. `data/evaluation/ratings.csv`: valoración final de J1, J2 y J3 por criterio;
   permanece vacía hasta recibir las calificaciones reales.
6. `data/telemetry/sensitivity_grid.csv`: resultados exportados de la malla
   factorial 5x5 del estudio H3.
7. `data/telemetry/turn_metrics.csv`: plantilla para una fila por turno con
   latencia, emisiones, entropía y fidelidad.
8. `src/codepysensibilidad.py` y `src/sensibilidad_h3_paralelo.py`: código
   fuente del motor y del estudio de sensibilidad.

La comparación usa dos condiciones sobre los mismos turnos: una línea base y
una versión con la directiva generada por el motor. La latencia y las
emisiones se estiman durante la ejecución con CodeCarbon; son referencias
dependientes del hardware y no mediciones eléctricas directas.

## Objetivos e hipótesis

El proyecto busca evaluar un chatbot local en español que combine
clasificación, contexto semántico y generación con un Small Language Model.
También busca medir coherencia, latencia, emisiones y estabilidad del motor.

El protocolo contrasta:

- **H1:** mejora de coherencia conversacional frente al baseline mediante cinco
  criterios evaluados por jueces ciegos.
- **H2:** sobrecarga menor de 500 ms y 1 gCO2eq por turno.
- **H3:** región compacta y contigua de estabilidad en `(alpha, c)`.

La evidencia disponible respalda la viabilidad técnica y la operación local
del prototipo. Los resultados sirven como una primera validación y dejan
abierta la ampliación del corpus, la telemetría y las comparaciones futuras.

## Procedimiento

1. Usar las exportaciones ya incluidas como referencia y no cambiar sus
   identificadores al incorporar nuevas versiones autorizadas.
2. Revisar que los encabezados coincidan con las plantillas y que no haya
   secretos, archivos personales, modelos binarios o temporales.
3. Comprobar los conteos declarados en `data/*/README.md`.
4. Instalar las dependencias de `requirements.txt` y ejecutar el código de
   `src/` desde la raíz del repositorio cuando se necesite reproducir el
   estudio.
5. Interpretar los resultados dentro del alcance del corpus, el hardware y
   los parámetros utilizados.

La fuente canónica del proyecto es
https://github.com/leogerm/chatbot-quantum-inspired-slm-sostenible.

La hoja de evaluación recibida contenía 80 registros y las dos respuestas por
turno. Se publicó como `data/evaluation/manual_jueces_ciegos.xlsx` después de
retirar sus puntuaciones de prueba. Por eso `data/evaluation/ratings.csv`
conserva solo su encabezado hasta recibir las calificaciones reales de J1, J2
y J3. No se usó `fill_judges.py`, ya que generaba puntuaciones sintéticas.
