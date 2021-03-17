import networkx as nx


class KeyTermGraph(object):
    def __init__(self, graph_yaml):
        self.graph = nx.read_yaml(graph_yaml)

    def get_neighbors_of(self, node_name: str):
        neighbors = [key for key in self.graph[node_name]]
        return neighbors

    def is_neighbor_of(self, node_name1: str, node_name2: str):
        neighbors = [key for key in self.graph[node_name1]]
        if node_name2 in neighbors:
            return True
        else:
            return False
