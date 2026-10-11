# chatbot-quantum-inspired-slm-sostenible

Chatbot híbrido quantum-inspired con Small Language Models locales para una
inteligencia artificial conversacional sostenible.

Repositorio público: https://github.com/leogerm/chatbot-quantum-inspired-slm-sostenible

## Evidencia experimental (D.3)

Este repositorio reserva una estructura mínima para poner a disposición la
evidencia del experimento sin publicar información personal ni artefactos
innecesarios:

- `data/experimental/`: corpus exportado de 80 turnos con categoría e
  identificador de diálogo, y las 160 respuestas de ambas condiciones.
- `data/evaluation/`: manual de evaluación ciega y calificaciones de los
  jueces J1, J2 y J3 por respuesta y criterio.
- `data/telemetry/`: registros por turno de latencia, emisiones, entropía y
  fidelidad.
- `src/`: código fuente reproducible, incluido `codepysensibilidad.py` y el
  estudio de sensibilidad H3.
- `REPRODUCIBILITY.md`: convención de archivos, controles y pasos de
  reproducción.

El manual publicado es una plantilla limpia para los jueces. Las puntuaciones
de prueba del archivo recibido no se publican como resultados experimentales.

## Estado de disponibilidad

| Evidencia | Ubicación | Estado |
| --- | --- | --- |
| Consultas, turnos y categorías | `data/experimental/` | 80 registros exportados |
| Respuestas de ambas condiciones | `data/experimental/` | 160 respuestas exportadas |
| Manual de jueces ciegos | `data/evaluation/manual_jueces_ciegos.xlsx` | 80 pares, listo para aplicar |
| J1/J2/J3 por criterio | `data/evaluation/ratings.csv` | Pendiente de calificaciones finales |
| Latencia, emisiones, entropía y fidelidad | `data/telemetry/` | Plantilla; matriz H3 disponible |
| Código fuente | `src/` | Motor y sensibilidad H3 integrados |

No se publican secretos, modelos binarios, archivos temporales, `__pycache__`
ni documentos personales.
