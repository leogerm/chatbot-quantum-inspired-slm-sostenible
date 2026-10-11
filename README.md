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
- `data/evaluation/`: calificaciones de los jueces J1, J2 y J3 por respuesta y
  criterio.
- `data/telemetry/`: registros por turno de latencia, emisiones, entropía y
  fidelidad.
- `src/`: código fuente reproducible. El archivo `codepysensibilidad.py` no
  está presente en este checkout; se incorporará aquí cuando esté disponible.
- `REPRODUCIBILITY.md`: convención de archivos, controles y pasos de
  reproducción.

Los CSV incluidos inicialmente son plantillas con encabezados: no contienen
resultados inventados. Los datos completos podrán añadirse respetando los
identificadores y las convenciones documentadas en cada carpeta.

## Estado de disponibilidad

| Evidencia | Ubicación | Estado |
| --- | --- | --- |
| Consultas, turnos y categorías | `data/experimental/` | 80 registros exportados |
| Respuestas de ambas condiciones | `data/experimental/` | 160 respuestas exportadas |
| J1/J2/J3 por criterio | `data/evaluation/` | Sin puntuaciones en la fuente recibida |
| Latencia, emisiones, entropía y fidelidad | `data/telemetry/` | Plantilla; matriz H3 disponible |
| Código fuente | `src/` | Código de sensibilidad integrado |

No se publican secretos, modelos binarios, archivos temporales, `__pycache__`
ni documentos personales.
