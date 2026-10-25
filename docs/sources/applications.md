# Applications

ARMxtend collects methods that the authors have developed and applied to real data, mostly in **healthcare**, in
**energy efficiency of buildings (IoT)** and in **law enforcement**. This page summarises those studies and says which ARMxtend modules cover the
methods they rely on, so you can follow the same workflow on your own data.

> The studies below were published before ARMxtend was released as a package. The "In ARMxtend" notes point to the
> modules that implement the same methods; they are not a claim that the published experiments were rerun with the
> package, and results obtained with it may differ in implementation details. Each study links to its DOI.

## Healthcare

### Fuzzy pattern mining of medical records: diagnostics and co-morbidity

Fernandez-Basso, Gutierrez-Batista, Morcillo-Jimenez, Vila & Martin-Bautista (2022),
[*Applied Soft Computing* 122, 108870](https://doi.org/10.1016/j.asoc.2022.108870).

Association rules and fuzzy logic are applied, in a distributed environment, to medical records, with the goal of
making the discovered patterns **understandable for end users**, not only accurate. The system
(a) preprocesses and enriches the database, (b) fuzzifies it to represent the uncertainty of real medical data,
(c) discovers relations between features such as diagnostics and hospital discharge, and (d) visualises the result.
It is shown on two health-related datasets.

*In ARMxtend:* the same four steps map to
[`preprocessing.FuzzyLib`](user_guide/preprocessing/FuzzyLib.md) (fuzzification),
[`FIM.BD_FARE`](user_guide/FIM/BD_FARE.md) and [`FIM.FARE`](user_guide/FIM/FARE.md) (distributed and sequential fuzzy
rule mining), and [`VizARM`](user_guide/VizARM.md) (visualisation).

### Behavioural patterns of patients from sensorised homes

Fernandez-Basso, Diaz-Jimenez, Lopez & Espinilla (2025),
[*Information Fusion* 123, 103307](https://doi.org/10.1016/j.inffus.2025.103307).

An automatic **fusion and fuzzification** methodology integrates multi-source sensor data into coherent fuzzy
variables, from which frequent behavioural patterns are discovered. It was validated in the sensorised homes of
patients with type II diabetes in Cabra (Cordoba, Spain), where it helps health professionals monitor and optimise
patient routines.

*In ARMxtend:* fuzzification with [`preprocessing.FuzzyLib`](user_guide/preprocessing/FuzzyLib.md) (including the
automatic partitions), then fuzzy itemset and rule mining with [`FIM.FARE`](user_guide/FIM/FARE.md).

### AIMDP: a data platform for hospital data silos

Ortega-Calvo, Morcillo-Jimenez, Fernandez-Basso, Gutierrez-Batista, Vila & Martin-Bautista (2023),
[*Future Generation Computer Systems* 143, 248-264](https://doi.org/10.1016/j.future.2023.02.002).

AIMDP (Artificial Intelligence Modern Data Platform) is a Big Data platform with AI-driven management, designed so
that users **without data-science expertise** can analyse their data, and with a layer for security and data
governance. It was used with two Spanish hospitals to analyse historical records that had been isolated in data silos
and never studied by researchers or clinicians. Among its analysis tools are the distributed association rule
algorithms collected in ARMxtend.

*In ARMxtend:* [`FIM.apriori`](user_guide/FIM/apriori.md), [`FIM.Eclat`](user_guide/FIM/Eclat.md) and
[`FIM.BD_ARE`](user_guide/FIM/BD_ARE.md) for crisp data, [`FIM.BD_FARE`](user_guide/FIM/BD_FARE.md) for fuzzy data.

### Level-by-level rule mining in clinical databases

Ortiz-Viso, Fernandez-Basso, Ruiz & Martin-Bautista (2025), *Lecture Notes in Networks and Systems* 1176, 143-154
([IPMU 2024](https://doi.org/10.1007/978-3-031-73997-2_13)).

A tiered, level-by-level approach to association rule mining on clinical databases, designed around their
hierarchical structure to support clinical decision-making.

### Federated fuzzy association rules *(related work, not in ARMxtend yet)*

Fernandez-Basso, Ruiz & Martin-Bautista (2025), *Lecture Notes in Networks and Systems* 1176, 131-142
([IPMU 2024](https://doi.org/10.1007/978-3-031-73997-2_12)).

A fuzzy association rule mining algorithm for **federated** settings, where each party keeps its data private. It aims
at protecting privacy and limiting communication overhead, with healthcare as a key application area. The federated
algorithm itself is not part of ARMxtend; the fuzzy rule mining it builds on is ([`FIM.FARE`](user_guide/FIM/FARE.md)).

## Security and law enforcement

### Knowledge extraction for police investigations

Fernandez-Basso, Gutierrez-Batista, Gomez-Romero, Ruiz & Martin-Bautista (2025),
[*Expert Systems* 42, e13524](https://doi.org/10.1111/exsy.13524).

An AI system that combines several tools to **assist police and law enforcement agencies** in an investigation: it
helps to collect, process and analyse large volumes of data, such as criminal activity on the dark web, and works as
an early warning and early action system that can point to possible crime hotspots. Association rule mining is one of
its knowledge-extraction tools. Its functionality is illustrated with data collected from the dark web, namely
advertisements offering firearms-related products.

*In ARMxtend:* [`ARM.association_rules`](user_guide/ARM/association_rules.md) and
[`FIM.FARE`](user_guide/FIM/FARE.md) to extract and filter the rules, and [`VizARM`](user_guide/VizARM.md) to explore
them.

## Energy efficiency of buildings (IoT)

Sensors in office buildings produce large volumes of consumption, occupancy and weather data. Several studies mine
them with fuzzy and Big Data association rules:

| Study | What it does | In ARMxtend |
|---|---|---|
| Fernandez-Basso, Ruiz & Martin-Bautista (2020), [*IEEE Trans. Fuzzy Systems*](https://doi.org/10.1109/TFUZZ.2020.2992180) | An automatic fuzzification method implemented with Big Data tools, used to detect interrelations among sensor and weather data from an office building. | [`preprocessing.FuzzyLib`](user_guide/preprocessing/FuzzyLib.md) |
| Fernandez-Basso, Ruiz & Martin-Bautista (2021), [*Int. J. Approximate Reasoning* 137](https://doi.org/10.1016/j.ijar.2021.07.004) | Spark algorithms for fuzzy association rules; evaluated, among other data, on sensor measurements from an office building. It shows that 10 equidistributed alpha-cuts are enough to find all the significant fuzzy rules. | [`FIM.BD_FARE`](user_guide/FIM/BD_FARE.md), [`FIM.Eclat`](user_guide/FIM/Eclat.md) |
| Ruiz, Gomez-Romero, Fernandez-Basso & Martin-Bautista (2022), [*IEEE Trans. Industrial Informatics* 18](https://doi.org/10.1109/TII.2021.3130052) | A Big Data architecture for building energy management, with components for data acquisition and for algorithms that analyse massive heterogeneous data; deployed in four buildings of different purposes. | the Spark algorithms run on top of an architecture like this one |
| Ruiz, Fernandez-Basso, Gomez-Romero & Martin-Bautista (2023), [*Scientific Reports* 13, 19810](https://doi.org/10.1038/s41598-023-47056-1) | An IoT system on a Big Data architecture with Spark association rule algorithms, which uncovered hidden patterns in the data of a building in Bucharest and helped improve its energy efficiency. | [`FIM.apriori`](user_guide/FIM/apriori.md), [`FIM.Eclat`](user_guide/FIM/Eclat.md), [`FIM.BD_ARE`](user_guide/FIM/BD_ARE.md) |

## The methods themselves

The papers that introduce the algorithms are listed in [Citing ARMxtend](cite.md): streaming frequent itemsets in
data streams (`SFIM`/`SARE`, 2019), Spark algorithms for crisp and fuzzy rules (2021, 2024), meta-association rules
over multiple datasets (2016) and the VizARE visualisation format (2026).

## Following the same workflow on your own data

| Your data | Typical pipeline |
|---|---|
| Sensors, IoT, smart buildings or homes | [`FuzzyLib`](user_guide/preprocessing/FuzzyLib.md) automatic fuzzification, [`FIM.BD_FARE`](user_guide/FIM/BD_FARE.md) (big data) or [`FIM.FARE`](user_guide/FIM/FARE.md), then [`VizARM`](user_guide/VizARM.md) |
| Clinical, administrative or investigative records | discretise or fuzzify, [`FIM.FARE`](user_guide/FIM/FARE.md) / [`ARM.association_rules`](user_guide/ARM/association_rules.md), prune with [`ARM.redundancy`](user_guide/ARM/redundancy.md), explore in [`VizARM`](user_guide/VizARM.md) |
| Massive transaction data | [`FIM.apriori`](user_guide/FIM/apriori.md) or [`FIM.Eclat`](user_guide/FIM/Eclat.md) on Spark, then [`FIM.BD_ARE`](user_guide/FIM/BD_ARE.md) (see [Scalability](user_guide/scalability.md)) |
| Data streams | [`SFIM` / `SARE`](user_guide/SARE_SFIM.md) |
| Rules from several datasets | [`ARM.meta_rules`](user_guide/ARM/meta_rules.md) |

The [examples](examples.md) include runnable scripts for each of these steps.
