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

## 7. Documentación

- [ ] Regenerar y publicar la web: `cd docs && python3 -m mkdocs gh-deploy` (yo no publico ni hago `push`).
- [ ] El código y los docstrings están en español y el README mezcla español e inglés. Decide el idioma de la
  documentación pública antes de la primera versión.
