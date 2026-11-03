# `datasets` -- datos sintéticos y pequeños conjuntos de ejemplo

Para probar un algoritmo, o para medirlo, hacen falta datos en los que se conozca la respuesta. `ARMxtend.datasets` tiene
generadores que **plantan patrones** (para poder comprobar que salen) y unos pocos conjuntos de datos diminutos dentro
del paquete; no se descarga nada.

```python
from ARMxtend.datasets import make_transactions, make_sequences, make_quantities, make_labeled_items, load_dataset

baskets, patterns = make_transactions(2000, n_items=60, n_patterns=8, pattern_length=3, random_state=7, return_patterns=True)
sequences = make_sequences(500, random_state=1)
quantities, profits = make_quantities(1000, random_state=1)     # for high_utility_rules
items, labels = make_labeled_items(1500, noise=0.05, random_state=3)   # for RuleClassifier
weather = load_dataset("weather")                             # weather.data, weather.target
```

## Generadores

Todos reciben `random_state` (un entero o un `numpy.random.Generator`) y dan los mismos datos para la misma semilla.

- **`make_transactions`**: al estilo del generador IBM Quest (Agrawal y Srikant, 1994, pero no es un port de él). Cada
  transacción es una mezcla de patrones plantados, cada ítem se pierde con probabilidad `corruption`, y se añaden ítems
  sueltos con una popularidad de tipo Zipf. Los pesos de los patrones son exponenciales, así que unos pocos son
  frecuentes y algunos raros. Con `return_patterns=True` obtiene también los patrones.
- **`make_sequences`**: secuencias de itemsets; con probabilidad 1/2 se intercala en orden un patrón plantado, cada uno
  de sus ítems en su propio elemento.
- **`make_quantities`**: cestas de `make_transactions` con cantidades en `1..max_quantity` y beneficios unitarios
  log-normales.
- **`make_labeled_items`**: una tabla booleana y una clase decidida por `n_rules` reglas de dos condiciones (se aplica la
  primera que cumple); `noise` cambia al azar esa fracción de las clases, de modo que alrededor de `1 - noise` es la
  mejor exactitud que cabe esperar.

## Conjuntos de ejemplo: `load_dataset(name)`

| nombre | contenido |
|---|---|
| `"han"` | 9 transacciones sobre 5 ítems (el ejemplo AllElectronics de Han y Kamber) |
| `"weather"` | 14 días, 4 atributos categóricos como ítems (`outlook=sunny`) y la clase `play` (la tabla del tenis de Quinlan) |
| `"sales"` | cantidades de 7 ítems en 5 ventas y sus beneficios unitarios (el ejemplo de [alta utilidad](../../user_guide/ARM/utility_rules.md)) |
| `"routines"` | 6 días de eventos de una casa, como secuencias |

El resultado es un `Dataset` (un diccionario cuyas claves se leen también como atributos) con `data`, `description` y,
según el caso, `target`, `profits` o `table`. Cada llamada da copias nuevas. `list_datasets()` lista los nombres.

Línea de comandos: [`armxtend generate`](../../user_guide/cli.md#generate) escribe datos sintéticos en los formatos que leen los demás
comandos.

## Qué conviene saber

- **Los datos sintéticos son más fáciles que los reales.** Los patrones plantados se separan limpiamente del ruido, lo
  que los datos reales rara vez hacen; use los generadores para comprobar y cronometrar algoritmos, no para comparar su
  calidad.
- **Algunos patrones plantados no se encontrarán** con un `min_supp` dado: los pesos exponenciales hacen que algunos sean
  raros (es deliberado). Compare con `return_patterns`, no con el número de patrones.
- **Los conjuntos pequeños se transcribieron de las tablas habituales de los libros de texto.** Compruébelos con las
  fuentes si publica cifras obtenidas con ellos.

## Referencias

- R. Agrawal, R. Srikant (1994). *Fast algorithms for mining association rules.* VLDB 1994, 487-499 (el generador de
  datos sintéticos usado para evaluar extractores de itemsets).
- J. Han, M. Kamber (2006). *Data Mining: Concepts and Techniques*, 2.ª ed. (el ejemplo de 9 transacciones).
- J. R. Quinlan (1986). *Induction of decision trees.* Machine Learning 1, 81-106 (la tabla del tiempo).
