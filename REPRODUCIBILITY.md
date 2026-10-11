# Reproducibilidad

## Alcance

El repositorio documenta la evidencia necesaria para el apartado D.3 sin
publicar datos que no estén disponibles en el checkout. Las plantillas CSV
describen la forma esperada de los datos y permanecen vacías hasta incorporar
los archivos originales o una exportación autorizada.

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

1. `data/experimental/queries.csv`: consultas exportadas con categoría e
   identificador de diálogo.
2. `data/experimental/turns.csv`: 80 turnos asociados a sus diálogos.
3. `data/experimental/responses.csv`: 160 respuestas, identificadas por
   condición.
4. `data/evaluation/ratings.csv`: valoración de J1, J2 y J3 por criterio.
5. `data/telemetry/turn_metrics.csv`: una fila por turno con latencia,
   emisiones, entropía y fidelidad.

## Procedimiento

1. Copiar las exportaciones autorizadas a las rutas anteriores sin cambiar los
   identificadores.
2. Revisar que los encabezados coincidan con las plantillas y que no haya
   secretos, archivos personales, modelos binarios o temporales.
3. Comprobar los conteos declarados en `data/*/README.md`.
4. Ejecutar el código de `src/` cuando esté disponible y conservar solo
   salidas reproducibles y pequeñas.

La fuente canónica del proyecto es
https://github.com/leogerm/chatbot-quantum-inspired-slm-sostenible.

La hoja de evaluación recibida contenía 80 registros y las dos respuestas por
turno. Se publicó como `data/evaluation/manual_jueces_ciegos.xlsx` después de
retirar sus puntuaciones de prueba. Por eso `data/evaluation/ratings.csv`
conserva solo su encabezado hasta recibir las calificaciones reales de J1, J2
y J3. No se usó `fill_judges.py`, ya que generaba puntuaciones sintéticas.
