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

### Hipótesis y lectura de la evidencia

- **H1 - coherencia:** se contrasta si el motor mejora la coherencia frente al
  baseline mediante cinco criterios y jueces ciegos. La campaña permite
  estudiar el efecto con un protocolo reproducible; la tesis no observa una
  mejora estadísticamente concluyente, por lo que el resultado se presenta
  como una base para mejorar el canal de transmisión de contexto.
- **H2 - costo:** se contrasta una sobrecarga menor de 500 ms y 1 gCO2eq por
  turno. Las estimaciones CodeCarbon son compatibles con un costo operativo
  bajo, aunque no demuestran por sí solas el umbral y no sustituyen una
  medición eléctrica directa.
- **H3 - estabilidad:** se estudia si existe una región compacta y contigua en
  `(alpha, c)`. La malla 5x5 cumple el criterio preregistrado sobre 23
  transiciones por celda, y deja una base clara para ampliar el análisis.

En conjunto, la evidencia apoya la viabilidad técnica, la inspección del
estado semántico y la operación local sostenible del prototipo. Las mejoras
prioritarias son publicar la telemetría cruda por turno y las calificaciones
finales, ampliar el corpus, comparar contra líneas base adicionales y probar
consultas con cambios temáticos abruptos.

## Evidencia experimental (D.3)

Este repositorio reserva una estructura mínima para poner a disposición la
evidencia del experimento sin publicar información personal ni artefactos
innecesarios:

- `data/experimental/`: corpus independiente de 57 consultas, 80 turnos con
  categoría e identificador de diálogo, y las 160 respuestas de ambas
  condiciones.
- `data/evaluation/`: manual de evaluación ciega y calificaciones de los
  jueces J1, J2 y J3 por respuesta y criterio.
- `data/telemetry/`: registros por turno de latencia, emisiones, entropía y
  fidelidad.
- `src/`: código fuente reproducible, incluido `codepysensibilidad.py` y el
  estudio de sensibilidad H3.
- `REPRODUCIBILITY.md`: convención de archivos, controles y pasos de
  reproducción.

El manual publicado es una plantilla limpia para los jueces. Las puntuaciones
de prueba recibidas para revisar el instrumento no se publican como resultados
experimentales.

## Correspondencia con la tesis

La tesis reporta una comparación pareada entre una condición base (A, modelo
con historial reciente y sin directiva) y una condición con el motor de
contexto (B), sobre los mismos 80 turnos. Reporta, entre otros resultados,
latencias medias de 18 200,22 ms (A) y 17 545,43 ms (B), emisiones medias de
0,010891 y 0,010537 gCO2eq por turno, y 1,7142 gCO2eq para las 160
generaciones. Estos valores son estimaciones instrumentadas con CodeCarbon,
específicas del hardware y la configuración descritos en la tesis.

La tesis también reporta una evaluación ciega de 80 pares por tres jueces,
con cinco criterios en escala 1-5, y una sensibilidad H3 de 25 configuraciones
con 23 transiciones por celda. El repositorio incluye el instrumento, el
código y la matriz H3; las calificaciones finales y la telemetría cruda por
turno deben incorporarse desde sus exportaciones autorizadas antes de
considerarse evidencia completa.

## Estado de disponibilidad

| Evidencia | Ubicación | Estado |
| --- | --- | --- |
| Consultas, turnos y categorías | `data/experimental/` | 57 consultas, 80 turnos exportados |
| Respuestas de ambas condiciones | `data/experimental/` | 160 respuestas exportadas |
| Manual de jueces ciegos | `data/evaluation/manual_jueces_ciegos.xlsx` | 80 pares, listo para aplicar |
| J1/J2/J3 por criterio | `data/evaluation/ratings.csv` | Pendiente de calificaciones finales |
| Latencia, emisiones, entropía y fidelidad | `data/telemetry/` | Plantilla; matriz H3 disponible |
| Código fuente | `src/` | Motor y sensibilidad H3 integrados |

No se publican secretos, modelos binarios, archivos temporales, `__pycache__`
ni documentos personales.
