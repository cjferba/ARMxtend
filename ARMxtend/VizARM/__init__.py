# -*- coding: utf-8 -*-
"""
VizARM: exportacion de reglas de asociacion (y, opcionalmente, de itemsets
frecuentes) a un grafo dirigido -- items como nodos, reglas como aristas --
en formatos estandar (GraphML, DOT) que pueden abrirse directamente en
herramientas como Gephi, Cytoscape o Graphviz.

@author Carlos Fernandez-Basso
"""
import csv

import networkx as nx


class AREtoGraph(object):
    """
    Construye un grafo dirigido de reglas de asociacion: un nodo por item, y
    una arista antecedente -> consecuente por cada regla, con las medidas de
    interes (confianza, lift, ...) como atributos de la arista.
    """

    def __init__(self, path=None, MeasuresRules=None, MeasuresItems=None,
                 SepRule=";", SepFI=";", SepItems=","):
        """
        Argumentos:
            path (str, opcional): ruta a un fichero CSV de reglas de
                asociacion, con columnas 'antecedents' y 'consequents'
                (items concatenados con `SepItems`), mas una columna por
                cada medida indicada en `MeasuresRules`. Si se omite, se
                construye un objeto vacio para poblarlo con `add_rules_from_dataframe`
            MeasuresRules (list[str], opcional): nombres de las columnas de
                medidas de las reglas (p.ej. ['confidence', 'lift']) que se
                incluiran como atributos de cada arista
            MeasuresItems (list[str], opcional): nombres de las columnas de
                medidas de los itemsets frecuentes (p.ej. ['support']), si
                se cargan por separado con `load_item_measures`
            SepRule (str): separador de campos del CSV de reglas
            SepFI (str): separador de campos del CSV de itemsets frecuentes
            SepItems (str): separador entre items dentro de un mismo lado
                (antecedente/consecuente) de una regla
        """
        self.MeasuresRules = MeasuresRules or []
        self.MeasuresItems = MeasuresItems or []
        self.SepRule = SepRule
        self.SepFI = SepFI
        self.SepItems = SepItems
        self.graph = nx.DiGraph()

        if path:
            self.load_rules(path)

    @classmethod
    def from_dataframe(cls, rules_df, rule_measures=("confidence", "lift")):
        """
        Construye el grafo directamente a partir de un pandas.DataFrame de
        reglas, en el mismo formato que devuelven `ARM.association_rules` o
        `FIM.FARE.fuzzy_association_rules` (columnas 'antecedents',
        'consequents' con frozensets de items, mas una columna por medida).

        Argumentos:
            rules_df (pandas.DataFrame): reglas de asociacion
            rule_measures (Iterable[str]): columnas de `rules_df` a incluir
                como atributos de cada arista

        Retorna:
            Una instancia de AREtoGraph con el grafo ya construido
        """
        instance = cls(MeasuresRules=list(rule_measures))
        instance.add_rules_from_dataframe(rules_df, rule_measures)
        return instance

    def add_rules_from_dataframe(self, rules_df, rule_measures=None):
        """Anade al grafo una regla por cada fila de `rules_df`."""
        measures = list(rule_measures) if rule_measures is not None else self.MeasuresRules
        for _, rule in rules_df.iterrows():
            antecedent = self.SepItems.join(sorted(rule["antecedents"]))
            consequent = self.SepItems.join(sorted(rule["consequents"]))
            attributes = {measure: rule[measure] for measure in measures if measure in rule}
            self._add_rule_edge(antecedent, consequent, attributes)

    def _add_rule_edge(self, antecedent, consequent, attributes):
        self.graph.add_node(antecedent)
        self.graph.add_node(consequent)
        self.graph.add_edge(antecedent, consequent, **attributes)

    def load_rules(self, path):
        """Carga reglas de asociacion desde un fichero CSV (ver `__init__`)."""
        with open(path, newline="") as ruleFile:
            reader = csv.DictReader(ruleFile, delimiter=self.SepRule)
            for row in reader:
                antecedent = row["antecedents"]
                consequent = row["consequents"]
                attributes = {measure: float(row[measure]) for measure in self.MeasuresRules if measure in row}
                self._add_rule_edge(antecedent, consequent, attributes)

    def load_item_measures(self, path):
        """
        Carga medidas de itemsets frecuentes (soporte, ...) desde un CSV con
        columna 'itemset' y una columna por cada medida en `MeasuresItems`,
        y las anade como atributos de los nodos correspondientes.
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

    def exportGraph(self, type=0):
        """
        Exporta el grafo construido.

        Argumentos:
            type (int): 0 -> cadena en formato GraphML (abrible en Gephi/
                Cytoscape); 1 -> cadena en formato DOT (Graphviz);
                2 -> el objeto networkx.DiGraph subyacente

        Retorna:
            Una cadena de texto (GraphML/DOT) o el grafo de networkx, segun
            `type`
        """
        if type == 0:
            return "\n".join(nx.generate_graphml(self.graph))
        elif type == 1:
            return self._to_dot()
        elif type == 2:
            return self.graph
        else:
            raise ValueError("type debe ser 0 (GraphML), 1 (DOT) o 2 (networkx.DiGraph), recibido: {}".format(type))

    def _to_dot(self):
        """Escritor DOT minimo, sin dependencias adicionales a networkx."""
        lines = ["digraph ARM {"]
        for node in self.graph.nodes:
            lines.append('    "{}";'.format(node))
        for source, target, attributes in self.graph.edges(data=True):
            attrText = ", ".join('{}="{}"'.format(key, value) for key, value in attributes.items())
            label = " [{}]".format(attrText) if attrText else ""
            lines.append('    "{}" -> "{}"{};'.format(source, target, label))
        lines.append("}")
        return "\n".join(lines)
