# Evaluación humana

`Evaluacion_Coherencia_Jueces_LLENA.xlsx` es la hoja de evaluación ciega con
las calificaciones de los tres evaluadores (pestañas Juez_1, Juez_2 y Juez_3):
80 pares de respuestas y cinco criterios por respuesta en escala de 1 a 5
(pertinencia, continuidad temática, corrección del contenido, adecuación de la
extensión y naturalidad del registro).

`ratings.csv` es la misma información en formato largo: 2 400 filas (3 jueces
x 160 respuestas x 5 criterios), con los jueces identificados como J1, J2 y J3.
Se genera con `python src/exportar_ratings.py`, que también reporta la
concordancia entre evaluadores (kappa ponderado cuadrático de 0,66, 0,60 y
0,65 por pares; promedio 0,64).

Los sufijos `-A` y `-B` de `response_id` corresponden a «Respuesta 1» y
«Respuesta 2» de la hoja, es decir, a la posición de presentación ciega.
