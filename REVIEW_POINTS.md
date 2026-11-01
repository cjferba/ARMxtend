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

## 8. Significación estadística (`ARM.significance`)

- [ ] **Referencias de memoria**: Holm (1979), Benjamini–Hochberg (1995), Benjamini–Yekutieli (2001), Westfall–Young
  (1993), Webb (2007), Wilson (1927). Verifica volúmenes y páginas antes de citar.
- [ ] **Tamaño de la familia de hipótesis.** Por defecto la corrección usa `rules.attrs["n_candidates"]`: todas las
  particiones antecedente/consecuente de los itemsets frecuentes (en las reglas temporales, solo las que cumplen la
  estructura temporal). Es una elección razonable y conservadora, pero no es exactamente la de Webb (2007), que cuenta
  el espacio de búsqueda completo; con `n_tests=` se puede cambiar. Decide si quieres otro valor por defecto.
- [ ] **FDR frente a FWER.** `fdr_bh` (por defecto) deja pasar en promedio hasta un 5 % de reglas falsas (en el ejemplo
  quedan 2 de 22). Para evitar cualquier falso positivo use `holm` o el test de permutación.
- [ ] **Test de dos colas** = doble del menor p-valor unilateral (no la definición de Fisher por probabilidades); la
  cola relevante para una regla es `greater`.
- [ ] **Permutación circular** (`null="shift"`): válida para dependencia temporal de cada serie, pero rompe también la
  relación de los items de calendario con las series (a propósito: es lo que se contrasta). No cubre tendencias
  comunes ni estacionalidad compartida si las series se generan de la misma fuente.
- [ ] **Reglas difusas y secuenciales**: no tienen test por regla; solo permutación con una medida (`lift`, ...) para las
  difusas. No hay test para reglas secuenciales.

## 9. Reglas en acción (`ARM.anomaly`, `ARM.prediction`, `ARM.classification`)

- [ ] **Referencias de memoria**: CBA (Liu, Hsu, Ma, KDD-98, pp. 80–86), Liu–Ma–Wong (PKDD 2000) y Bayardo–Agrawal–
  Gunopulos (DMKD 4, 2000, pp. 217–240). Verifica títulos y páginas antes de citar.
- [ ] **Peso de una violación = `-log2(1 - confianza)`** con suelo `floor=1e-3`. Es una elección mía (bits de sorpresa),
  no un estándar; la suma de pesos sobre reglas correlacionadas cuenta de más (por eso se recomienda podar). Decide si
  prefieres otro peso por defecto.
- [ ] **Anomalía = violar una regla fiable.** Una regla del 99 % se viola sola el 1 % de las veces: esas violaciones
  naturales son indistinguibles de los fallos (en el ejemplo, 9 de 39 instantes). El umbral lo decide la aplicación.
- [ ] **Predicción con reglas secuenciales**: dicen "después", no "inmediatamente", así que `target="future"` es el criterio
  coherente y `target="next"` el exigente. Las reglas no pueden vetarse entre sí (el ejemplo sigue prediciendo `door`
  tras `coffee`). El baseline de popularidad es el de los eventos más frecuentes aún no vistos.
- [ ] **CBA**: es CBA-CB M1 (sin la fase de reglas con soporte mínimo múltiple completo del CBA 2.0). Los soportes por clase
  son una simplificación de Liu–Ma–Wong. La lista suele empezar por reglas de la clase mayoritaria con confianza 1.
- [ ] **`prune_unproductive_rules`**: la mejora se compara con todas las subreglas presentes en el conjunto; si las reglas
  generales no se minaron (umbral alto), no se detecta la improductividad.
- [ ] **CLI**: `anomalies`/`forecast` reservan por filas (`--train-fraction`), no por fechas; `classify` baraja con semilla.
  El JSON de fechas pasó a ISO (antes, milisegundos desde 1970) para evitar el aviso de pandas 4.

## 10. Secuencias difusas (`FIM.fuzzy_sequential`, `ARM.fuzzy_sequential_rules`)

- [ ] **Referencia de memoria**: Fiot, Laurent & Teisseire (2007), *From crispness to fuzziness: three algorithms for
  soft sequential pattern mining*, IEEE Trans. Fuzzy Systems 15(6), 1263-1277, y Klement, Mesiar & Pap (2000),
  *Triangular Norms*. Comprueba autores, título, volumen y páginas. **No implemento sus algoritmos**: la definición
  (grado de un patrón = mejor encaje, con t-norma `min` o producto; soporte = media sobre las secuencias) es mía, y
  así lo dice la documentación. Decide si te vale o si prefieres la definición de ese artículo (que fuzzifica también
  el tiempo).
- [ ] **Solo elementos de un ítem.** Los patrones son listas de ítems sueltos (`<x1 x2 ...>`); no hay elementos con
  varios ítems simultáneos como en `FIM.sequential`. Con grados 0/1 coinciden con los patrones de `prefixspan` de un
  ítem por elemento (comprobado en los tests).
- [ ] **Grado de una regla** = `max_k T(mu_X(k), mu_Y(k))` con `mu_X`/`mu_Y` como t-norma de los mejores grados de
  cada ítem hasta / después de `k`. Con grados 0/1 reproduce `sequential_rules` (comprobado). Es la extensión más
  natural que se me ocurre, pero no hay una definición canónica; contrástala con tu criterio.
- [ ] **La confianza difusa no es comparable con la nítida** (en el ejemplo, 0.63 frente a 0.82-1.00 según el umbral
  nítido): cuenta los días parcialmente. Para ordenar reglas, `lift`. No hay test de significación para estas reglas.
- [ ] **Coste.** Se construye un tensor denso secuencias x posición x ítems (los grados 0 no se ahorran): para bases muy
  grandes o con muchas etiquetas hay que bajar `max_len`, subir `min_supp` o usar `min_degree`. No hay versión Spark.
- [ ] **CLI** (`fuzzy-sequences`, `fuzzy-sequential-rules`): la fuzzificación usa `FuzzyLib.AutoFuzzification` con
  3 etiquetas por cuantiles y triangulares por defecto, y `--min-degree 0` (con triangulares casi toda lectura tiene
  algún grado en la etiqueta vecina: usa 0.1-0.3). Sin `--sequence-column` todo el fichero es una sola secuencia.

## 11. Taxonomías y reglas multinivel (`FIM.taxonomy`, `ARM.generalized_rules`)

- [ ] **Referencias de memoria**: Srikant & Agrawal (1995), *Mining generalized association rules*, VLDB, pp. 407–419, y
  Han & Fu (1995), *Discovery of multiple-level association rules from large databases*, VLDB, pp. 420–431. Comprueba
  títulos y páginas. No tuve acceso a los textos.
- [ ] **Definición de regla R-interesante reconstruida de memoria.** Implemento: sin antecesoras, o, respecto a **todas**
  sus antecesoras *cercanas* que a su vez sean interesantes, soporte **o** confianza reales >= R veces los esperados
  (fórmulas en el docstring). Dudo de dos matices del artículo: (a) si exige estar por encima de *cada* antecesora cercana o
  solo de una; (b) si además pide que el itemset sea R-interesante. Las antecesoras se buscan **solo en el conjunto de
  reglas dado**: si la regla general no pasó `min_threshold`, la específica se considera sin antecesoras (y se queda).
  Contrástalo con el artículo antes de decir que es "el" criterio de Srikant–Agrawal.
- [ ] **Soporte por nivel: elección propia.** Un itemset necesita el MAYOR de los mínimos de los niveles de sus items
  (el del item más general). Es lo que conserva la propiedad antimonótona con la poda de Apriori, pero **no** es el
  esquema de Han & Fu (que minan cada nivel por separado, con el umbral de ese nivel, y cruzan niveles con otras
  variantes). Consecuencia a decidir: una regla `Jacket => Footwear` necesita el umbral del nivel 1 aunque `Jacket` sea
  del nivel 3, así que con `{1: 0.2, 3: 0.04}` no salen reglas producto→categoría de soporte intermedio.
- [ ] **No es Cumulate completo.** Solo aplico la poda de itemsets con un item y su antecesor; no filtro los antecesores
  añadidos a las transacciones a los que aparecen en algún candidato (otra optimización del artículo). El minero es
  por niveles con vectores booleanos por itemset (memoria ~ transacciones x itemsets frecuentes del nivel más ancho);
  para bases grandes con muchos itemsets frecuentes puede consumir mucha memoria. No hay versión Spark.
- [ ] **Niveles en taxonomías con varios padres.** El nivel de un item es 1 + el mayor nivel de sus padres (así un
  antecesor siempre tiene nivel menor, que es lo que usa el orden de proceso de `interesting_rules`); con caminos de
  longitud distinta, dos antecesores al mismo "nivel" pueden estar a distancias distintas de la raíz. `at_level` devuelve el
  antecesor más profundo de nivel <= L en cada camino.
- [ ] **Significación.** `significant_rules` funciona (la familia es `n_candidates`), pero las reglas generalizadas están
  muy correlacionadas (una regla y las de sus categorías), así que la corrección BH/Holm es conservadora o laxa según el
  caso; no está estudiado.
- [ ] **CLI** (`generalized-rules`): la taxonomía es un CSV `hijo,padre` con cabecera y el mismo `--delimiter` que las
  cestas; `--level-supp` fija el soporte de los niveles >= 2 (`--min-supp` es el del nivel 1).

## 12. Reglas cuantitativas (`ARM.quantitative_rules`)

- [ ] **Referencia de memoria**: Srikant & Agrawal (1996), *Mining quantitative association rules in large relational
  tables*, SIGMOD, pp. 1–12. Comprueba título y páginas. **La fórmula de `partial_completeness_intervals`**
  (`2n / (m (K - 1))` intervalos por atributo) también es de memoria: verifícala en el artículo (el valor que da, 160 para
  `n=2, m=0.05, K=1.5`, es muy alto y el ejemplo usa 10).
- [ ] **No es el algoritmo del artículo.** Fijo los rangos de antemano (todas las secuencias de intervalos base contiguos
  con soporte <= `max_supp`) y los trato como una taxonomía por inclusión, de modo que todo el minado y la poda
  R-interesante son los de `generalized_rules`; el artículo los combina sobre la marcha. El conjunto de rangos y los
  soportes son los mismos (comprobado contra enumeración exhaustiva), pero el coste es O(intervalos²) ítems por columna y
  una fila pertenece a todos los rangos que contienen su valor: con 20+ intervalos y muchas columnas la matriz booleana
  de ítems crece rápido.
- [ ] **Poda R-interesante**: valen las dudas de la sección 11 (antecesoras cercanas, "soporte o confianza"). En el
  artículo cuantitativo la definición tiene matices propios (p. ej. exigir que también el itemset sea interesante) que no
  recuerdo; contrástala antes de citar "el criterio de Srikant–Agrawal".
- [ ] **`max_supp`** descarta los rangos de varios intervalos con soporte mayor, pero un intervalo base se queda aunque
  lo supere (empates en los valores). **Intervalos base de igual frecuencia** (`quantile`): con valores repetidos salen
  menos de `n_intervals`; los límites se escriben con 6 cifras significativas (`{:g}`), así que dos límites que difieran
  más allá de la sexta cifra darían la misma etiqueta (improbable).
- [ ] **Qué es numérico.** Una columna numérica con <= `n_intervals` valores distintos (p. ej. una valoración de 1 a 5)
  se trata como categórica (`col=3`) y pierde el orden; fuerza el tratamiento con `quantitative=[...]`. Los valores
  ausentes no generan ítem. El signo menos de los límites es U+2212 (los nombres de ítem no admiten `-`), lo que puede
  sorprender al exportar o filtrar por texto.
- [ ] **`quantitative_rules` ajusta la partición con los mismos datos que mina** (los intervalos dependen de los datos
  completos). Para validar fuera de muestra, ajusta un `IntervalPartition` con un periodo y pásalo con `partition=`.
- [ ] **Comparación con las reglas difusas**: el ejemplo usa un solo conjunto de datos sintéticos con una región plantada
  que no coincide con las etiquetas difusas; no es una comparación general (con límites suaves o etiquetas con significado
  propio, las difusas pueden ser mejores). Mide con tus datos.
- [ ] **CLI** (`quantitative-rules`): umbral de confianza por defecto 0.8, `--max-supp` 0.5, 8 intervalos.

## 13. Reglas top-k, raras y de alta utilidad (`FIM.topk`, `FIM.rare`, `FIM.utility`, `ARM.topk_rules`, `ARM.rare_rules`, `ARM.utility_rules`)

- [ ] **Referencias de memoria**: Fournier-Viger & Tseng (2012), *Mining top-k association rules*, Canadian AI, pp. 61–73;
  Szathmary, Napoli & Valtchev (2007), *Towards rare itemset mining*, IEEE ICTAI, pp. 305–312; Yao, Hamilton & Butz (2004),
  *A foundational approach to mining itemset utilities from databases*, SIAM SDM, pp. 482–486; Liu, Liao & Choudhary (2005),
  *A two-phase algorithm for fast discovery of high utility itemsets*, PAKDD, pp. 689–695; y la mención a Sahoo et al.
  (2015) en el docstring de `utility_rules.py`. Comprueba títulos, congresos y páginas antes de citarlos.
- [ ] **Top-k no es el algoritmo del artículo.** `top_k_rules` enumera itemsets el mejor primero y los parte en reglas;
  TopKRules de Fournier-Viger y Tseng sube el umbral mientras expande reglas. El resultado es exacto (comprobado contra la
  enumeración exhaustiva) pero el coste no es el suyo: en datos densos con `k` grande la frontera de la cola puede crecer
  mucho en memoria. `max_len=6` por defecto recorta las reglas largas; decide si ese valor es razonable.
- [ ] **Los filtros (`min_conf`, `min_lift`, `max_antecedent`, `max_consequent`) se aplican antes de ordenar**, de modo que
  salen las `k` mejores que los cumplen. Sin filtro de lift, lo más frecuente es lo más obvio (en el ejemplo, `fruit => bread`
  con lift 1.00). Los p-values de `significant_rules` sobre un top-k son optimistas (las reglas se eligieron por frecuentes):
  la familia es `n_candidates` (las examinadas), pero no es una corrección por la selección.
- [ ] **Raras.** Los MRI de un solo ítem no dan reglas; los superconjuntos de un MRI no se generan, así que una combinación
  rara que necesita un tercer ítem solo aparece si sus pares son frecuentes. `max_supp` decide qué es frecuente y por tanto qué
  es minimal. Los umbrales por defecto (`min_count=2` en las reglas, `1` en los itemsets) son decisión mía. Las raras entre
  ítems independientes tienen lift ≈ 1 y solo `min_lift` o la significación las separan.
- [ ] **Alta utilidad.** Algoritmo de dos fases simple y exacto, no el más rápido (HUI-Miner, EFIM, etc.): matrices densas
  transacciones × ítems en memoria y candidatos por niveles; con `min_utility` bajo y muchos ítems puede ser lento. Los
  beneficios y las cantidades deben ser ≥ 0 (con negativos la cota TWU no vale). No hay versión Spark. La fracción
  `min_utility` depende de toda la base.
- [ ] **Las reglas de utilidad usan las medidas clásicas** (cuentan transacciones) más `utility` y `utility_share`; todas las
  particiones de un itemset tienen la misma utilidad, y una regla puede ser de alta utilidad solo porque contiene un ítem muy
  valioso (en el ejemplo, `cheese => bread` con lift 0.99): sin `min_lift` llena la lista. Las definiciones de «confianza de
  utilidad» de la literatura (utilidad de X∪Y entre la de X) no se usan.
- [ ] **CLI** (`top-k-rules`, `rare-rules`, `high-utility-rules`): `-k` y `--max-supp` son obligatorios; `--max-len 0` quita el
  límite de `top-k-rules`; los beneficios van en un CSV `item,beneficio` con cabecera (sin él, 1 para todos). Las tres admiten
  `--alpha` y las demás opciones de significación. Un `n_candidates` pequeño (reglas examinadas) hace la corrección por pruebas
  múltiples más laxa de lo que sería sobre todas las reglas posibles.

## 14. Spark (DECLAT, entrada DataFrame, secuencias)

- [ ] **Reescritura de `DECLAT`/`FuzzyDECLAT`**: resultado idéntico al anterior en 100 000 transacciones (comparado clave a clave y valor a valor) y a FP-Growth en tests aleatorios. Cambio de comportamiento menor: un item repetido en una transacción difusa cuenta con su mayor grado (antes, uno arbitrario). Los TIDs son int32 si caben.
- [ ] **Sigue acotado por el driver**: las TID-lists se recogen y se difunden; ahora ocupan ~4 (+8) bytes por aparición. No es un Eclat distribuido.
- [ ] **Todas las medidas son `local[n]`** en una sola máquina, una ejecución mejor de dos; la tabla de `scalability.md` se regeneró solo para data, cores y sequences (items y cache conservan las cifras anteriores).
- [ ] **Secuencias**: el PrefixSpan local gana por debajo de ~100 000 secuencias; el de Spark solo compensa con datos mayores. Entrada DataFrame con `array<array<string>>`; los items se ordenan y deduplican por elemento.
- [ ] **`FIM.spark_io`**: los items no pueden contener `,` ni `-`; una fila nula es una transacción vacía.

## 15. Documentación

- [ ] Regenerar y publicar la web: `cd docs && python3 -m mkdocs gh-deploy` (yo no publico ni hago `push`).
- [ ] El código y los docstrings están en español y el README mezcla español e inglés. Decide el idioma de la
  documentación pública antes de la primera versión.
