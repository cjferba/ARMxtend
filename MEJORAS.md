# Lista de mejoras

Numerada para ir tachando (~~hecho~~). Las que dependen de una decisión tuya o de una acción fuera del repositorio
llevan **[tuyo]**.

## Para publicar
1. **[tuyo]** Decidir la licencia: CC BY-NC 4.0 no es una licencia de software aprobada por la OSI.
2. **[tuyo]** Crear los entornos de GitHub `pypi` y `testpypi`, registrar el publisher en pypi.org y test.pypi.org, y hacer la release `v0.1.0`.
3. **[tuyo]** Activar Zenodo al publicar la release y añadir el DOI a `CITATION.cff`.
4. **[tuyo]** SPMF en la comparativa (necesita descargar su `.jar`; pídemelo).
5. **[tuyo]** Decidir el idioma de la documentación (código en español, web en inglés).
6. Verificar las referencias bibliográficas escritas de memoria (`REVIEW_POINTS` §13, §15, §17).
7. **[tuyo]** Borrar los `ARMxtend/FIM/BDARE/Result_reglas*.text` sueltos y confirmar las descripciones de Applications.

## Rendimiento
8. ~~Optimizar `FFIM.fpgrowth` y `ARM.association_rules` (a la par de mlxtend).~~
9. ~~Minero denso por bits (`FIM.dense_itemsets`), más rápido que el Apriori de mlxtend con una tabla booleana.~~
10. Eclat distribuido de verdad (hoy `DECLAT`/`FuzzyDECLAT` recogen las TID-lists en el driver).
11. Medir en un clúster Spark real (todo es `local[n]`).
12. ~~Minero denso difuso (`FIM.dense_fuzzy_itemsets`): mismos itemsets que `fuzzy_fpgrowth`, de 45x a 280x más rápido.~~

## Investigación
13. ~~Significación de reglas difusas, regla a regla (`ARM.fuzzy_significant_rules`).~~
14. ~~Contrastar la `FConf` agregada por alpha-cortes (`statistic="fconf"` en `fuzzy_rule_significance`).~~
15. ~~Reglas secuenciales con veto (`ARM.sequential_veto_rules`, `X => ~Y`).~~
16. ~~Elementos con varios items simultáneos en los patrones secuenciales difusos (`itemset_elements=True`).~~
17. Fuzzificar también el tiempo (Fiot et al.).
18. ~~Significación para reglas secuenciales (`ARM.sequential_rule_significance`, test de permutación del orden).~~
19. ~~CBA con soportes mínimos múltiples completos (`min_supp`/`min_conf` por clase y `min_count`).~~
20. ~~Clasificación con reglas difusas (`ARM.fuzzy_class_rules`, `FuzzyRuleClassifier`).~~
21. Reglas sobre flujos: `rule_drift` incremental.
22. ~~Utilidad de la regla (no solo del itemset): `utility_confidence` y `consequent_utility_share`.~~
23. Explicabilidad: por qué se dispara una regla y comparación con alternativas.
24. Reglas sobre grafos o datos relacionales.

## Calidad del proyecto
25. Cuadernos para más temas: difusas, secuenciales, Spark.
26. Integración continua con tests, mypy y un benchmark de regresión de rendimiento.
27. Guía de contribución más completa.
28. Datos reales de ejemplo, además de los sintéticos.
29. Publicar la web automáticamente en cada release.
