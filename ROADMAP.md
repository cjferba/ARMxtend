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
| 2 | **Reglas en acción** (`ARM.anomaly`, `ARM.prediction`, `ARM.classification`): detección de anomalías por violación de reglas; predicción del siguiente instante/evento con reglas temporales y secuenciales; reglas de asociación de clase (CBA); poda de reglas improductivas | hecho |
| 3 | **Patrones y reglas secuenciales difusos** (`FIM.fuzzy_sequential`, `ARM.fuzzy_sequential_rules`): eventos con grado de pertenencia, t-normas `min` / producto | hecho |
| 4 | **Reglas multinivel / con taxonomía** (`FIM.taxonomy`, `ARM.generalized_rules`): jerarquías de ítems, soporte por nivel, reglas R-interesantes que podan las redundantes entre niveles | hecho |
| 5 | **Reglas cuantitativas** (`ARM.quantitative_rules`): discretización por intervalos con fusión (Srikant–Agrawal) frente a la difusa | hecho |
| 6 | **Reglas raras, de alta utilidad y top-k** (`FIM.topk`, `FIM.rare`, `FIM.utility`, `ARM.topk_rules`, `ARM.rare_rules`, `ARM.utility_rules`): las k mejores sin umbral, itemsets raros minimales, itemsets y reglas de alta utilidad | hecho |
| 7 | **Spark**: entrada DataFrame/Parquet; DECLAT/FuzzyDECLAT con TID-lists compactas; benchmark de secuencias | hecho |
| 8 | **API estilo scikit-learn**, generadores sintéticos y conjuntos de datos de ejemplo (`estimators`, `datasets`, `armxtend generate`) | hecho |
| 9 | **Proyecto**: `CITATION.cff` / Zenodo, cuadernos, comparativas con SPMF y mlxtend, idioma de la documentación | parcial: hechos `CITATION.cff`, cuaderno y comparativa con mlxtend; faltan Zenodo, SPMF y decidir el idioma |

## Notas por paso

**1. Significación.** Entregado en `ARMxtend/ARM/significance.py`. Queda fuera: test por regla para reglas difusas y
secuenciales (solo permutación con una medida para las difusas).

**2. Reglas en acción.** Entregado: (a) `anomaly_scores` / `detect_anomalies` puntúan cada fila por las reglas fiables
que viola; (b) `rule_scores` / `forecast_next` / `predict_next_events` anticipan con reglas temporales y secuenciales y
`evaluate_*` mide el acierto frente a una línea base; (c) `RuleClassifier` (CBA-CB M1). Queda fuera: reglas
secuenciales con veto (una regla que descarte eventos), CBA con soportes mínimos múltiples completos, y clasificación
con reglas difusas.

**3. Secuencias difusas.** Entregado: `fuzzy_prefixspan` (PrefixSpan con poda antimonótona exacta sobre un tensor de
grados), `fuzzy_pattern_degree`, `fuzzy_sequences_from_memberships` (de una tabla de pertenencias, p. ej. la de
`FuzzyLib`, a secuencias), `fuzzy_sequential_rules` / `fuzzy_rule_degree` y los comandos `fuzzy-sequences` /
`fuzzy-sequential-rules`. Queda fuera: elementos con varios ítems simultáneos, versión Spark, fuzzificar también el
tiempo (Fiot et al.) y tests de significación para reglas difusas.

**4. Taxonomías.** Entregado: `Taxonomy` (varios padres, niveles, `at_level`), `extend_transactions`,
`generalize_transactions`, `generalized_itemsets` (sin itemsets con un ítem y su antecesor, soporte por nivel),
`generalized_rules`, `interesting_rules` (R-interesantes) y el comando `generalized-rules`. Queda fuera: taxonomías
difusas (pertenencia parcial a una categoría), reglas multinivel sobre datos difusos o secuencias, y una versión Spark.

**5. Cuantitativas.** Entregado: `IntervalPartition` (intervalos base de igual frecuencia o anchura, rangos hasta `max_supp`,
`transform` para datos nuevos), `quantitative_rules` (sobre la maquinaria de taxonomías, con grupos excluyentes por columna),
`partial_completeness_intervals` y el comando `quantitative-rules`. Queda fuera: el algoritmo original de combinación sobre la
marcha, reglas con rangos disjuntos (p. ej. dos tramos de edad), selección automática de `n_intervals`, y la comparación
sistemática con las difusas en conjuntos reales.

**6. Raras, alta utilidad, top-k.** Entregado: `top_k_itemsets` / `top_k_rules` (búsqueda el mejor primero sobre el árbol de
enumeración, exacta, con parada perezosa), `minimal_rare_itemsets` / `rare_rules` (Apriori por niveles sobre los
frecuentes, con la base compacta de los raros minimales), `high_utility_itemsets` / `high_utility_rules` (dos fases con
la utilidad ponderada por transacción), la representación vertical en bits compartida (`FIM/_vertical.py`) y los
comandos `top-k-rules`, `rare-rules` y `high-utility-rules`. Todas las salidas siguen el formato de `association_rules`
(más `utility` / `utility_share`) y llevan `attrs` (`n_transactions`, `n_candidates`), de modo que pasan por
`significant_rules` y por la CLI sin cambios. Queda fuera: versión Spark, beneficios negativos, otras definiciones de
confianza de utilidad, algoritmos de utilidad más rápidos (HUI-Miner, EFIM), top-k con medidas distintas del soporte y
reglas con ítems raros sueltos.

**7. Spark.** Medido antes con `benchmarks/spark_scalability.py`. Entregado: TID-lists como arrays numpy (resultado idéntico bit a bit), difusión única,
`FIM.spark_io` (DataFrame/Parquet), `prefixspan_spark` sin `collect` y el experimento `sequences`. Sigue siendo un diseño acotado por la
memoria del driver (se recogen las TID-lists), solo que unas 100 veces menor; un Eclat realmente distribuido (por clases de equivalencia) queda fuera, así
como medir en un cluster real.

**9. Proyecto.** Las decisiones abiertas de publicación (entornos `pypi`/`testpypi`, fecha del 0.1.0, licencia CC BY-NC
frente a una OSI) están en `REVIEW_POINTS.md`, sección 6.
