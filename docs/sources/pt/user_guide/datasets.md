# `datasets` -- dados sintéticos e pequenos conjuntos de exemplo

Para testar um algoritmo, ou medi-lo, são necessários dados em que a resposta seja conhecida. `ARMxtend.datasets` tem
geradores que **plantam padrões** (para que se possa verificar que eles aparecem) e alguns conjuntos de dados minúsculos
dentro do pacote; nada é baixado.

```python
from ARMxtend.datasets import make_transactions, make_sequences, make_quantities, make_labeled_items, load_dataset

baskets, patterns = make_transactions(2000, n_items=60, n_patterns=8, pattern_length=3, random_state=7, return_patterns=True)
sequences = make_sequences(500, random_state=1)
quantities, profits = make_quantities(1000, random_state=1)     # for high_utility_rules
items, labels = make_labeled_items(1500, noise=0.05, random_state=3)   # for RuleClassifier
weather = load_dataset("weather")                             # weather.data, weather.target
```

## Geradores

Todos recebem `random_state` (um inteiro ou um `numpy.random.Generator`) e dão os mesmos dados para a mesma semente.

- **`make_transactions`**: no estilo do gerador IBM Quest (Agrawal e Srikant, 1994, mas não é um port dele). Cada
  transação é uma mistura de padrões plantados, cada item é perdido com probabilidade `corruption`, e itens avulsos são
  sorteados com popularidade do tipo Zipf. Os pesos dos padrões são exponenciais, então alguns são frequentes e outros
  raros. Com `return_patterns=True` você também obtém os padrões.
- **`make_sequences`**: sequências de itemsets; com probabilidade 1/2 um padrão plantado é intercalado em ordem, cada um
  dos seus itens em seu próprio elemento.
- **`make_quantities`**: cestas de `make_transactions` com quantidades em `1..max_quantity` e lucros unitários
  log-normais.
- **`make_labeled_items`**: uma tabela booleana e uma classe decidida por `n_rules` regras de duas condições (vale a
  primeira que se aplica); `noise` muda ao acaso essa fração das classes, de modo que cerca de `1 - noise` é a melhor
  acurácia que se pode esperar.

## Conjuntos de exemplo: `load_dataset(name)`

| nome | conteúdo |
|---|---|
| `"han"` | 9 transações sobre 5 itens (o exemplo AllElectronics de Han e Kamber) |
| `"weather"` | 14 dias, 4 atributos categóricos como itens (`outlook=sunny`) e a classe `play` (a tabela do tênis de Quinlan) |
| `"sales"` | quantidades de 7 itens em 5 vendas e seus lucros unitários (o exemplo de [alta utilidade](../../user_guide/ARM/utility_rules.md)) |
| `"routines"` | 6 dias de eventos de uma casa, como sequências |

O resultado é um `Dataset` (um dicionário cujas chaves também são lidas como atributos) com `data`, `description` e,
conforme o caso, `target`, `profits` ou `table`. Cada chamada dá cópias novas. `list_datasets()` lista os nomes.

Linha de comando: [`armxtend generate`](../../user_guide/cli.md#generate) escreve dados sintéticos nos formatos que os outros comandos
leem.

## O que convém saber

- **Dados sintéticos são mais fáceis que dados reais.** Os padrões plantados se separam nitidamente do ruído, o que os
  dados reais raramente fazem; use os geradores para verificar e cronometrar algoritmos, não para comparar a sua
  qualidade.
- **Alguns padrões plantados não serão encontrados** com um dado `min_supp`: os pesos exponenciais tornam alguns deles
  raros (é de propósito). Compare com `return_patterns`, não com o número de padrões.
- **Os conjuntos pequenos foram digitados a partir das tabelas habituais dos livros-texto.** Confira-os com as fontes se
  publicar números obtidos com eles.

## Referências

- R. Agrawal, R. Srikant (1994). *Fast algorithms for mining association rules.* VLDB 1994, 487-499 (o gerador de
  dados sintéticos usado para avaliar extratores de itemsets).
- J. Han, M. Kamber (2006). *Data Mining: Concepts and Techniques*, 2.ª ed. (o exemplo de 9 transações).
- J. R. Quinlan (1986). *Induction of decision trees.* Machine Learning 1, 81-106 (a tabela do tempo).
