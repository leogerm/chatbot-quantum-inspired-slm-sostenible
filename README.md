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

### Hipótesis y lectura general

- **H1 - coherencia:** se contrasta si el motor mejora la coherencia frente al
  baseline mediante cinco criterios y jueces ciegos. El resultado sirve como
  punto de partida para mejorar la transmisión de contexto.
- **H2 - costo:** se contrasta una sobrecarga menor de 500 ms y 1 gCO2eq por
  turno. Las estimaciones disponibles son compatibles con una operación local
  de bajo costo.
- **H3 - estabilidad:** se estudia si existe una región compacta y contigua en
  `(alpha, c)`. La malla 5x5 muestra un comportamiento estable y deja una
  base clara para ampliar el análisis.

En conjunto, la evidencia apoya la viabilidad técnica, la inspección del
estado semántico y la operación local sostenible del prototipo. El proyecto
queda abierto a mejoras en datos, medición y comparación experimental.

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
