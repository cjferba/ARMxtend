# Hoja de ruta de ARMxtend

Secuencia ordenada de mejoras posteriores a la 0.1.0. Cada paso se implementa, se prueba, se documenta (guía de
usuario, ejemplo, CLI si procede, CHANGELOG y `REVIEW_POINTS.md`) y se confirma con un commit propio antes de pasar al
siguiente. El estado se marca aquí al terminar cada paso.

Criterio de orden: primero lo que hace **fiables** las reglas ya existentes (significación), después lo que las pone **a
trabajar** (anomalías, predicción, clasificación), después **nuevos tipos de regla**, después **escala** y, por último,
**proyecto** (citación, cuadernos, comparativas).

| # | Paso | Estado |
|---|---|---|
| 1 | **Significación estadística** (`ARM.significance`): test de Fisher / chi², Bonferroni / Holm / FDR, intervalos de confianza, permutación (minP) con desplazamientos circulares para series | hecho |
| 2 | **Reglas en acción**: detección de anomalías por violación de reglas; predicción del siguiente evento con reglas temporales y secuenciales; reglas de asociación de clase (tipo CBA) | pendiente |
| 3 | **Patrones y reglas secuenciales difusos** (cantidades/intensidades en los eventos) | pendiente |
| 4 | **Reglas multinivel / con taxonomía** (jerarquías de ítems, poda de reglas redundantes entre niveles) | pendiente |
| 5 | **Reglas cuantitativas**: discretización por intervalos con fusión (Srikant–Agrawal) frente a la difusa | pendiente |
| 6 | **Reglas raras, de alta utilidad y top-k** | pendiente |
| 7 | **Spark**: entrada DataFrame/Parquet; reescritura de FuzzyDECLAT/DECLAT sin recoger en el driver; benchmark de secuencias | pendiente |
| 8 | **API estilo scikit-learn**, generadores sintéticos y conjuntos de datos de ejemplo | pendiente |
| 9 | **Proyecto**: `CITATION.cff` / Zenodo, cuadernos, comparativas con SPMF y mlxtend, idioma de la documentación | pendiente |

## Notas por paso

**1. Significación.** Entregado en `ARMxtend/ARM/significance.py`. Queda fuera: test por regla para reglas difusas y
secuenciales (solo permutación con una medida para las difusas).

**2. Reglas en acción.** Tres piezas independientes: (a) puntuar cada transacción/ventana por las reglas fuertes que
viola (reutiliza `ARM.exception_rules`); (b) dado un historial, proponer el evento siguiente con las reglas
temporales/secuenciales que se cumplen, evaluado con una partición temporal; (c) un clasificador por reglas de clase
(consecuente = clase) con la cobertura de CBA, evaluado frente a una línea base.

**3–6.** Cada uno es un módulo nuevo en `ARM/` o `FIM/` con el mismo patrón que los existentes: función pura sobre
`DataFrame`, columnas de medidas estándar y `attrs` con el contexto (`n_transactions`, ...), de modo que las reglas
resultantes pasen por `significant_rules`, `prune_redundant_rules` y la CLI sin cambios.

**7. Spark.** Antes de tocar nada, medir con `benchmarks/spark_scalability.py` para no optimizar a ciegas.

**9. Proyecto.** Las decisiones abiertas de publicación (entornos `pypi`/`testpypi`, fecha del 0.1.0, licencia CC BY-NC
frente a una OSI) están en `REVIEW_POINTS.md`, sección 6.
