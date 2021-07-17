from typing import List

import pandas as pd
import networkx as nx


def create_keyterm_graph(filepath: str, cleaner=None) -> nx.Graph:
            
    """create an nx.Graph object to represent the relations between keyterms

    :param filepath: a excel file contain at least 2 columns with the first 2 columns `keyterm` and `synonyms`
    :type filepath: str
    :return: keyterm graph
    :rtype: nx.Graph
    """


    if cleaner is None:
        # return a do-nothing cleaner for default mode
        cleaner = lambda text: text
        
    df = pd.read_excel(filepath, header=0)
    df = df.iloc[:, :2]  # assume it only contains 2 columns
    df.columns = ['keyterm', 'synonyms']
    df = df.astype(str)

    # print(df.head())

    graph = nx.Graph()

    for row in df.itertuples():
        keyterm = cleaner(row.keyterm)
        synonyms = [cleaner(synonym) for synonym in row.synonyms.split(',') if synonym]  # assume comma-seperate
        
        if keyterm not in graph:
            graph.add_node(keyterm)
        
        for synonym in synonyms:
            # iterate through the synonyms
            if synonym not in graph:
                # if not in the graph, add a new node for this synonym
                graph.add_node(synonym)
            for synonym_ in synonyms:
                # iterate through the the rest of synonyms
                if synonym_ == synonym:
                    # if it is itself, add an edge with the keyterm
                    graph.add_edge(keyterm, synonym_)
                else:
                    # otherwise add an edge between 2 synonyms
                    if synonym_ not in graph:
                        graph.add_node(synonym_)
                    graph.add_edge(synonym_, synonym)
    
    return graph



class KeyTermGraph(object):
    def __init__(self, graph_yaml: str):
        """representation of key terms and theire relations

        :param graph_yaml: configuration file of the key term graph
        :type graph_yaml: str
        """
        self.graph = nx.read_yaml(graph_yaml)

    def get_neighbors_of(self, node_name: str) -> List[str]:
        """get neighbors of a key term

        :param node_name: key term name
        :type node_name: str
        :return: list of related key terms
        :rtype: List[str]
        """
        neighbors = [key for key in self.graph[node_name]]
        return neighbors

    def is_neighbor_of(self, node_name1: str, node_name2: str) -> bool:
        """check whether two key terms are related

        :param node_name1: first key term 
        :type node_name1: str
        :param node_name2: second key term
        :type node_name2: str
        :return: whether two key terms are related
        :rtype: bool
        """
        neighbors = [key for key in self.graph[node_name1]]
        if node_name2 in neighbors:
            return True
        else:
            return False

