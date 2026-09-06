# -*- coding: utf-8 -*-
"""
VizARM: transformacion de reglas de asociacion a un grafo dirigido tipado,
siguiendo la metodologia VizARE de:

    Fernandez-Basso, C., Ruiz, M.D., Molina-Solana, M., Martin-Bautista, M.J.
    (2026). "VizARE: An Intermediate Representation to Support the
    Visualization of Association Rules in Data Mining". Future Internet,
    18(7), 374. https://doi.org/10.3390/fi18070374

A diferencia de un grafo item-a-item (donde una arista uniria directamente
cada item del antecedente con cada item del consecuente, perdiendo la
estructura de la regla en cuanto tiene mas de un item por lado), VizARE
modela el grafo con DOS tipos de nodo -- item y rule -- y las conecta
mediante aristas dirigidas con un rol semantico explicito: antecedent
(item -> rule) y consequent (rule -> item). Esto permite representar sin
ambiguedad reglas con varios items en el antecedente y/o el consecuente
(Seccion 4.1, Algoritmo 1 del articulo), y guarda las medidas de interes
(soporte, confianza, lift, ...) como atributos del propio nodo regla, no de
la arista.

Ademas del grafo tipado, este modulo implementa:
    * La capa de resumen opcional (Seccion 4.2): nodos "summary" que agrupan
      reglas equivalentes segun una firma (mismo antecedente, mismo
      consecuente, o el par antecedente->consecuente), agregando sus
      medidas (cuenta, min, max, media, cuantiles) para poder explorar
      grandes conjuntos de reglas sin perder la trazabilidad a las reglas
      originales (`members`).
    * Exportacion al JSON Graph Format (JGF), el formato intermedio
      propuesto en el articulo (Seccion 3.3.3 y 4), ademas de GraphML y DOT
      (formatos ya soportados por herramientas como Gephi/Cytoscape/
      Graphviz/NetworkX, tambien mencionadas en la Seccion 4.4 del articulo).

@author Carlos Fernandez-Basso
"""
import csv
import json

import networkx as nx
import numpy as np

from ..ARM.meta_rules import rule_key

_VALID_SIGNATURES = ("antecedent", "consequent", "antecedent_consequent")


def _default_item_group(item):
    """
    Grupo por defecto de un item: el nombre del atributo si el item sigue
    la convencion 'atributo_etiqueta' (Seccion 4.1: "Item nodes encode
    atomic attribute-value pairs using the normalized syntax
    attribute_value"), la misma que usan las columnas generadas por
    `preprocessing.FuzzyLib.Fuzzification`; si no, el propio item.
    """
    return item.split("_", 1)[0] if "_" in item else item


def _sanitize_for_export(value):
    """GraphML/DOT no admiten listas como valor de atributo; las serializamos como texto."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return ";".join(str(v) for v in value)
    return value


class AREtoGraph(object):
    """
    Construye el grafo tipado (item/rule/summary) de un conjunto de reglas
    de asociacion, siguiendo la metodologia VizARE (ver modulo).
    """

    def __init__(self, path=None, MeasuresRules=None, MeasuresItems=None,
                 SepRule=";", SepFI=";", SepItems=","):
        """
        Argumentos:
            path (str, opcional): ruta a un fichero CSV de reglas de
                asociacion, con columnas 'antecedents' y 'consequents'
                (items concatenados con `SepItems`), mas una columna por
                cada medida indicada en `MeasuresRules`. Si se omite, se
                construye un objeto vacio para poblarlo con `add_rule`/
                `add_rules_from_dataframe`
            MeasuresRules (list[str], opcional): nombres de las medidas de
                las reglas (p.ej. ['support', 'confidence', 'lift']) que se
                incluiran como atributos de cada nodo 'rule'
            MeasuresItems (list[str], opcional): nombres de las medidas de
                los itemsets frecuentes (p.ej. ['support']), si se cargan
                por separado con `load_item_measures`
            SepRule (str): separador de campos del CSV de reglas
            SepFI (str): separador de campos del CSV de itemsets frecuentes
            SepItems (str): separador entre items dentro de un mismo lado
                (antecedente/consecuente) de una regla
        """
        self.MeasuresRules = list(MeasuresRules) if MeasuresRules else ["support", "confidence"]
        self.MeasuresItems = list(MeasuresItems) if MeasuresItems else []
        self.SepRule = SepRule
        self.SepFI = SepFI
        self.SepItems = SepItems
        self.graph = nx.DiGraph()

        if path:
            self.load_rules(path)

    @classmethod
    def from_dataframe(cls, rules_df, rule_measures=("support", "confidence", "lift"), item_group_fn=None):
        """
        Construye el grafo directamente a partir de un pandas.DataFrame de
        reglas, en el mismo formato que devuelven `ARM.association_rules` o
        `FIM.FARE.fuzzy_association_rules` (columnas 'antecedents',
        'consequents' con frozensets de items, mas una columna por medida).

        Argumentos:
            rules_df (pandas.DataFrame): reglas de asociacion
            rule_measures (Iterable[str]): columnas de `rules_df` a incluir
                como atributos de cada nodo 'rule'
            item_group_fn (callable[str] -> str, opcional): ver `add_rule`

        Retorna:
            Una instancia de AREtoGraph con el grafo ya construido
        """
        instance = cls(MeasuresRules=list(rule_measures))
        instance.add_rules_from_dataframe(rules_df, rule_measures, item_group_fn=item_group_fn)
        return instance

    def add_rules_from_dataframe(self, rules_df, rule_measures=None, item_group_fn=None):
        """Anade al grafo una regla (Algoritmo 1) por cada fila de `rules_df`."""
        measures = list(rule_measures) if rule_measures is not None else self.MeasuresRules
        for _, rule in rules_df.iterrows():
            antecedent = sorted(rule["antecedents"])
            consequent = sorted(rule["consequents"])
            measureValues = {measure: float(rule[measure]) for measure in measures if measure in rule}
            self.add_rule(antecedent, consequent, measureValues, item_group_fn=item_group_fn)

    def add_rule(self, antecedent, consequent, measures=None, rule_id=None, item_group_fn=None):
        """
        Algoritmo 1 (RulesToGraph) de Fernandez-Basso et al. (2026): anade
        una regla como un nodo 'rule' (con sus medidas de interes como
        atributos propios) conectado mediante aristas dirigidas a los nodos
        'item' de su antecedente (relation='antecedent') y su consecuente
        (relation='consequent').

        Argumentos:
            antecedent (Iterable[str]): items del antecedente de la regla
            consequent (Iterable[str]): items del consecuente de la regla
            measures (dict[str, float], opcional): medidas de interes de la
                regla (soporte, confianza, lift, factor de certeza, ...),
                almacenadas como atributos del nodo 'rule'
            rule_id (str, opcional): identificador del nodo de regla; por
                defecto se genera con `ARM.meta_rules.rule_key`
            item_group_fn (callable[str] -> str, opcional): funcion que
                calcula el atributo 'group' de un nodo item (por defecto,
                `_default_item_group`: el prefijo antes de '_', como en
                'temperature_cold' -> 'temperature')

        Retorna:
            El identificador del nodo de regla anadido
        """
        antecedent = sorted(antecedent)
        consequent = sorted(consequent)
        measures = dict(measures) if measures else {}
        item_group_fn = item_group_fn or _default_item_group

        if rule_id is None:
            rule_id = "rule::{}".format(rule_key(antecedent, consequent))

        self.graph.add_node(
            rule_id, kind="rule", name="{} -> {}".format(
                self.SepItems.join(antecedent), self.SepItems.join(consequent)),
            group="rule", **measures)

        for item in antecedent:
            self._add_item_node(item, item_group_fn)
            self.graph.add_edge(item, rule_id, relation="antecedent")
        for item in consequent:
            self._add_item_node(item, item_group_fn)
            self.graph.add_edge(rule_id, item, relation="consequent")

        return rule_id

    def _add_item_node(self, item, item_group_fn):
        if item not in self.graph:
            self.graph.add_node(item, kind="item", name=item, group=item_group_fn(item))

    def summarize(self, signature="antecedent_consequent", measures=None, quantiles=(0.25, 0.5, 0.75)):
        """
        Capa de resumen (Seccion 4.2 del articulo): anade al grafo un nodo
        'summary' por cada grupo de reglas equivalentes bajo la firma
        indicada, agregando sus medidas de interes. Los nodos de resumen
        se conectan a los MISMOS nodos item, con el mismo sentido semantico
        (antecedent/consequent) que los nodos de regla, por lo que el grafo
        resultante contiene ambas capas (regla a regla y resumida) para que
        la herramienta de visualizacion elija la granularidad.

        Argumentos:
            signature (str): 'antecedent' (agrupa por el mismo conjunto de
                items antecedente), 'consequent' (mismo consecuente) o
                'antecedent_consequent' (mismo par antecedente->consecuente,
                por defecto)
            measures (Iterable[str], opcional): medidas de los nodos 'rule'
                a agregar (por defecto, `self.MeasuresRules`)
            quantiles (Iterable[float]): cuantiles adicionales a calcular
                por medida (Seccion 4.2: "optional quantiles")

        Retorna:
            list[str]: los identificadores de los nodos de resumen creados
        """
        if signature not in _VALID_SIGNATURES:
            raise ValueError("signature debe ser uno de {}, recibido: {}".format(_VALID_SIGNATURES, signature))
        measures = list(measures) if measures is not None else self.MeasuresRules

        groups = {}
        for node, data in self.graph.nodes(data=True):
            if data.get("kind") != "rule":
                continue

            antecedentItems = tuple(sorted(
                source for source, _, edgeData in self.graph.in_edges(node, data=True)
                if edgeData.get("relation") == "antecedent"))
            consequentItems = tuple(sorted(
                target for _, target, edgeData in self.graph.out_edges(node, data=True)
                if edgeData.get("relation") == "consequent"))

            if signature == "antecedent":
                key = antecedentItems
            elif signature == "consequent":
                key = consequentItems
            else:
                key = (antecedentItems, consequentItems)

            group = groups.setdefault(key, {
                "ruleIds": [], "antecedent": antecedentItems, "consequent": consequentItems})
            group["ruleIds"].append(node)

        summaryIds = []
        for index, group in enumerate(groups.values()):
            summaryId = "summary::{}::{}".format(signature, index)
            stats = {"rule_count": len(group["ruleIds"])}

            for measure in measures:
                values = [self.graph.nodes[ruleId][measure] for ruleId in group["ruleIds"]
                          if measure in self.graph.nodes[ruleId]]
                if not values:
                    continue
                values = np.asarray(values, dtype=float)
                stats["{}_min".format(measure)] = float(values.min())
                stats["{}_max".format(measure)] = float(values.max())
                stats["{}_mean".format(measure)] = float(values.mean())
                for quantile in quantiles:
                    stats["{}_q{}".format(measure, int(round(quantile * 100)))] = float(np.quantile(values, quantile))

            self.graph.add_node(
                summaryId, kind="summary", name="summary({})[{}]".format(signature, len(group["ruleIds"])),
                group="summary", signature=signature, members=list(group["ruleIds"]), **stats)

            for item in group["antecedent"]:
                self.graph.add_edge(item, summaryId, relation="antecedent")
            for item in group["consequent"]:
                self.graph.add_edge(summaryId, item, relation="consequent")

            summaryIds.append(summaryId)

        return summaryIds

    def load_rules(self, path):
        """
        Carga reglas de asociacion desde un CSV con columnas 'antecedents' y
        'consequents' (items separados por `SepItems`), mas una columna por
        cada medida de `MeasuresRules`, anadiendolas al grafo con `add_rule`.
        """
        with open(path, newline="") as ruleFile:
            reader = csv.DictReader(ruleFile, delimiter=self.SepRule)
            for row in reader:
                antecedent = row["antecedents"].split(self.SepItems)
                consequent = row["consequents"].split(self.SepItems)
                measures = {measure: float(row[measure]) for measure in self.MeasuresRules if measure in row}
                self.add_rule(antecedent, consequent, measures)

    def load_item_measures(self, path):
        """
        Carga medidas de itemsets individuales (p.ej. soporte) desde un CSV
        con columnas 'itemset' y una por cada medida de `MeasuresItems`,
        y las anade como atributos de los nodos item correspondientes.
        """
        with open(path, newline="") as itemFile:
            reader = csv.DictReader(itemFile, delimiter=self.SepFI)
            for row in reader:
                itemset = row["itemset"]
                if itemset not in self.graph:
                    continue
                for measure in self.MeasuresItems:
                    if measure in row:
                        self.graph.nodes[itemset][measure] = float(row[measure])

    def to_jgf_dict(self):
        """
        Serializa el grafo al formato intermedio propuesto en el articulo:
        JSON Graph Format (JGF, ver Seccion 3.3.3 y 4.1), como un dict de
        Python listo para `json.dumps` o para insertar directamente en una
        base de datos documental (MongoDB) o de grafos (Neo4j), tal y como
        se describe en la Seccion 4.3.

        Retorna:
            dict con la estructura {'graph': {'directed': True, 'nodes':
            {id: {'label', 'metadata': {...}}}, 'edges': [{'source',
            'target', 'relation', 'metadata'}]}}
        """
        nodes = {}
        for node, data in self.graph.nodes(data=True):
            metadata = {key: value for key, value in data.items() if key not in ("name",)}
            nodes[node] = {"label": data.get("name", node), "metadata": metadata}

        edges = [
            {"source": source, "target": target, "relation": data.get("relation"), "metadata": {}}
            for source, target, data in self.graph.edges(data=True)
        ]

        return {"graph": {"directed": True, "type": "association-rules", "nodes": nodes, "edges": edges}}

    def to_jgf(self, indent=2):
        """Serializa el grafo a una cadena JSON en formato JGF (ver `to_jgf_dict`)."""
        return json.dumps(self.to_jgf_dict(), indent=indent)

    def exportGraph(self, type=0):
        """
        Exporta el grafo construido.

        Argumentos:
            type (int): 0 -> cadena JGF (JSON Graph Format, el formato
                intermedio propuesto por Fernandez-Basso et al., 2026,
                exportable a MongoDB/Neo4j o a librerias como D3.js/Bokeh);
                1 -> cadena en formato GraphML (Gephi/Cytoscape/NetworkX);
                2 -> cadena en formato DOT (Graphviz); 3 -> el objeto
                networkx.DiGraph subyacente

        Retorna:
            Una cadena de texto (JGF/GraphML/DOT) o el grafo de networkx,
            segun `type`
        """
        if type == 0:
            return self.to_jgf()
        elif type == 1:
            return "\n".join(nx.generate_graphml(self._sanitized_graph()))
        elif type == 2:
            return self._to_dot()
        elif type == 3:
            return self.graph
        else:
            raise ValueError(
                "type debe ser 0 (JGF), 1 (GraphML), 2 (DOT) o 3 (networkx.DiGraph), recibido: {}".format(type))

    def _sanitized_graph(self):
        """Copia del grafo con atributos de tipo lista serializados como texto (GraphML no admite listas)."""
        sanitized = self.graph.copy()
        for _, data in sanitized.nodes(data=True):
            for key, value in list(data.items()):
                data[key] = _sanitize_for_export(value)
        return sanitized

    def _to_dot(self):
        """Escritor DOT minimo, sin dependencias adicionales a networkx."""
        lines = ["digraph ARM {"]
        for node, data in self.graph.nodes(data=True):
            label = data.get("name", node)
            lines.append('    "{}" [kind="{}", label="{}"];'.format(node, data.get("kind", ""), label))
        for source, target, attributes in self.graph.edges(data=True):
            attrText = ", ".join(
                '{}="{}"'.format(key, _sanitize_for_export(value)) for key, value in attributes.items())
            label = " [{}]".format(attrText) if attrText else ""
            lines.append('    "{}" -> "{}"{};'.format(source, target, label))
        lines.append("}")
        return "\n".join(lines)
