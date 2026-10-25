# Puntos de revisión

Lista de lo que conviene que **revises tú** antes de citar o publicar cada módulo. Son los sitios donde no pude
contrastar el código con el texto de la fuente original (no tenía el paper) o donde tomé una decisión de diseño
que afecta a resultados científicos. Todo lo demás está cubierto por tests automáticos.

Cuando hayas revisado un punto, bórralo de aquí o márcalo con `[x]`.

## 1. Fidelidad a los papers

- [ ] **Reglas de excepción y anómalas** (`ARM.exception_rules`). No tuve el paper de Delgado, Ruiz & Sánchez (2011,
  IJUFKS). Implementé las condiciones de Suzuki sobre soporte y confianza y lo dejé indicado en el docstring y en la
  documentación. Contrástalo con el paper antes de citarlo como implementación fiel.
- [ ] **Reglas graduales** (`ARM.gradual_rules`). El soporte por pares de filas sigue la idea de Hüllermeier, pero no
  lo validé contra su texto. Usa memoria `O(n²)`: para tablas grandes hay que pasar `max_rows`.
- [ ] **Reglas con ítems ausentes** (`ARM.negative_rules`). Sigue la sección 4 de Delgado et al. (la ausencia es el
  complemento por niveles, no `1 - μ`) y, por defecto, solo niega ítems cuyo positivo es frecuente (restricción de
  Han–Beheshti). Comprueba que esa restricción por defecto es la que quieres.
- [ ] **Referencias de `docs/sources/cite.md`.** Corregí de memoria el orden de autores y el título de Bastide et al.
  y quité el título de Han–Beheshti (solo queda "refs. [18, 19]" de Delgado). Verifica las referencias completas
  antes de publicar.

## 2. Medidas de interés

- [ ] En las medidas nuevas (`jaccard`, `cosine`, `kulczynski`, `all_confidence`, `max_confidence`, `added_value`,
  `zhangs_metric`, `yules_q`) un cociente 0/0 devuelve `0`. Solo la confianza difusa usa la convención del paper
  (0/0 := 1). Decide si quieres unificar el criterio.
- [ ] **Factor de certeza.** Lo alineé con la Eq. (7) del paper: con el antecedente ausente vale `0` (antes `-1`).
  Afecta sobre todo a los α-cortes altos de `FIM.FARE` (el FCF difuso salía sesgado a la baja); está en el `CHANGELOG`.

## 3. FIMoTS (`SARE` / `SFIM`)

Al verificar el algoritmo de streaming contra un oráculo de fuerza bruta (tras cada deslizamiento de ventana) aparecieron
varios fallos y los corregí. Son cambios en tu algoritmo de investigación, así que revísalos:

- [ ] `FIMoTS_Tree`: nuevos `getParentKey`, `getRightSiblings`, `getLeftSiblings`; `deleteChildNodes` ahora borra
  **todos** los descendientes (antes solo los hijos directos).
- [ ] `FIMoTS_Algorithm`: nueva extensión de los hermanos izquierdos (`FIMoTS_ExtendLeftSiblings`) y seguimiento de
  ítems nuevos (`FIMoTS_AddNewItems`, llamado al final de `FIMoTS_Main`). Son una reconstrucción mía de lo que el
  algoritmo debe hacer para que el árbol contenga todos los itemsets frecuentes de la ventana; confirma que coincide
  con tu descripción en el paper.
- [ ] `FIMoTS_Initial` ya no vuelve a registrar hijos que existen en el árbol (antes duplicaba contadores).
- [ ] Modo `TEXTFILE`: los ficheros se leen ahora con la ruta completa y en orden alfabético.
- [ ] `SFIM` es ahora un alias de `SARE` (los 8 subpaquetes reexportan desde `SARE`) en lugar de una copia
  byte a byte. Si pensabas darle lógica propia, hay que volver a separarlos.

Verificación: 23 tests contra Spark real y 60 escenarios aleatorios adicionales, todos sin diferencias con la fuerza
bruta. Detalle en `docs/sources/user_guide/SARE_SFIM.md`.

## 4. Spark y escalabilidad

- [ ] Los benchmarks (`benchmarks/spark_scalability.py`) se ejecutaron en **una sola máquina** (`local[n]`). Miden
  el reparto entre núcleos y el crecimiento del coste con los datos, **no** la red ni el comportamiento en un clúster
  real. No hay medidas en clúster; no cites esos números como "escalabilidad en clúster".
- [ ] `DECLAT` y `FuzzyDECLAT` llevan las TID-lists al driver (`groupByKey().collectAsMap()`): el límite de datos lo
  pone la memoria del driver. Está medido en el benchmark; decide si quieres reescribirlo.
- [ ] El broadcast de candidatos en `DApriori` ahora se lee dentro del closure (antes viajaba el valor entero en
  cada tarea). Mismo resultado, pero conviene que confirmes que no esperabas el comportamiento anterior.

## 5. VizARE interactivo (`ARMxtend.VizARM`)

- [ ] `to_html` genera un visor autocontenido (sin CDN). Lo probé en el navegador integrado de la app; pruébalo en
  tu navegador habitual y con tus propios conjuntos de reglas (cientos o miles de reglas).
- [ ] `to_cypher` y `to_documents` (Neo4j / MongoDB) están probados como texto/estructuras, **no** contra una
  instancia real de Neo4j ni de MongoDB.
- [ ] Compara el esquema del grafo (nodos item/rule/summary, aristas antecedent/consequent) con el del paper VizARE
  (Future Internet, 2026) para confirmar que es el que quieres exponer.

## 6. Publicación en PyPI

El código y los workflows están preparados, pero **no se ha publicado nada** (necesita tu cuenta de PyPI).

- [ ] Registrar el *pending publisher* en PyPI (y opcionalmente TestPyPI): proyecto `ARMxtend`, propietario `cjferba`,
  repositorio `ARMxtend`, workflow `release.yml`, entorno `pypi`. Pasos en `docs/sources/CONTRIBUTING.md`.
- [ ] Crear los entornos `pypi` y `testpypi` en los ajustes del repositorio (con revisor obligatorio en `pypi`).
- [ ] Comprobar que el nombre `ARMxtend` está libre en PyPI y fijar la versión inicial (`__version__` en
  `ARMxtend/__init__.py`, ahora `0.1.0`) antes de crear el *release* `v<versión>`.
- [ ] Licencia: el paquete declara CC BY-NC 4.0 (no es una licencia aprobada por la OSI). PyPI la acepta, pero
  `pip` y las herramientas de cumplimiento la mostrarán como no libre.

## 7. Series temporales, secuencias y deriva de reglas

Módulos nuevos: `preprocessing.timeseries`, `ARM.temporal_rules`, `ARM.rule_drift`, `FIM.sequential`,
`ARM.sequential_rules`. No tuve acceso a los papers (las editoriales bloquean la descarga), así que lo que sigue es
lo que hay que contrastar antes de citarlos como implementaciones fieles.

- [ ] **Referencias escritas de memoria** (en `cite.md` y en los docstrings): Lu, Han & Feng (1998); Tung, Lu, Han &
  Feng (2003, TKDE 15(1), 43-56); Lin, Keogh, Wei & Lonardi (2007, DMKD 15, 107-144); Pei et al. (2004, TKDE 16(11),
  1424-1440, con Jianyong Wang entre los autores); Fournier-Viger et al. (2015, TKDE 27(8), 2203-2216) y RuleGrowth
  (2011, mencionado solo en el docstring de `sequential_rules.py`). Comprueba títulos, volúmenes, páginas y autores.
- [ ] **Reglas temporales** (`ARM.temporal_rules`). Uso retardos **relativos al instante actual** y fuerzo que el
  consecuente esté en el instante actual (lag 0); el soporte se mide sobre las `N - max_lag` filas con todos los
  retardos. Lu et al. formulan reglas inter-transacción con dimensiones y ventanas más generales: la mía es la
  especialización habitual para series y sensores. Decide si te vale o si quieres consecuentes en otros instantes.
- [ ] **Coste y redundancia.** Con `max_lag` grande y muchos símbolos el número de itemsets crece mucho y salen muchas
  variantes de una misma regla; en el ejemplo uso `prune_redundant_rules`. No hay poda específica de reglas
  temporales (p. ej. por antecedente mínimo).
- [ ] **SAX sin PAA.** `symbolize(method="sax")` normaliza y corta con cuantiles de la normal, punto a punto, sin la
  agregación por tramos (PAA) del artículo original. Está documentado, pero no es SAX completo.
- [ ] **Reglas secuenciales** (`ARM.sequential_rules`). Uso la definición de regla válida de Fournier-Viger et al.
  (todos los ítems de X antes que todos los de Y, dentro de X y de Y el orden no cuenta) con soporte = secuencias
  donde es válida / secuencias y confianza = válidas / secuencias que contienen X. **No es RuleGrowth**: enumero
  itemsets frecuentes (FP-Growth sobre las secuencias como cestas) y pruebo sus particiones. Da el mismo resultado
  (comprobado contra enumeración exhaustiva) pero no tiene la eficiencia de RuleGrowth. Confirma que el
  denominador de la confianza es el del artículo.
- [ ] **PrefixSpan** (`FIM.sequential.prefixspan`). Comprobado contra enumeración exhaustiva en datos aleatorios, y
  `prefixspan_spark` (PrefixSpan de MLlib) contra la versión local en un Spark real, pero solo con bases pequeñas.
  No hay benchmark de escalabilidad de las secuencias.
- [ ] **Deriva de reglas** (`ARM.rule_drift`). Los estados `stable/emerging/vanishing/intermittent`, el umbral
  `stable_fraction=0.8` y la pendiente (recta de mínimos cuadrados sobre las ventanas con la regla) son decisiones
  mías, sin contraste estadístico: una regla "emergente" puede ser ruido de una ventana corta. No sustituye a las
  tendencias de FIMoTS (incrementales sobre Spark): es la versión por lotes en una máquina.
- [ ] **CLI** (`temporal-rules`, `rule-drift`, `sequences`, `sequential-rules`). Convenciones que he elegido: las
  columnas con solo 0/1 son eventos, las demás numéricas se simbolizan en 3 símbolos por cuantiles, y el formato
  de secuencias en texto es `a,b;c;d` (elementos separados por `;`). Compruébalas con tus datos reales.

## 8. Documentación

- [ ] Regenerar y publicar la web: `cd docs && python3 -m mkdocs gh-deploy` (yo no publico ni hago `push`).
- [ ] El código y los docstrings están en español y el README mezcla español e inglés. Decide el idioma de la
  documentación pública antes de la primera versión.
