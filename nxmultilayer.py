import networkx as nx
from copy import deepcopy
from itertools import product
"""
Work with multilayer networks on top of NetworkX
"""

class node_dict(dict):
    """Helper class to ensure aspects are updated correctly for new nodes"""
    def __init__(self, mg):
        super().__init__()
        self._mg = mg

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == len(self._mg.aspects):
            super().__setitem__(key, value)
            for k, a in zip(key, self._mg.aspects):
                if k not in a:
                    a[k] = dict()
        else:
            raise ValueError("node key does not match number of layers")


class adj_dict(dict):
    """Helper class to ensure all new nodes have correct format"""
    def __init__(self, mg):
        super().__init__()
        self._mg = mg

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == len(self._mg.aspects):
            super().__setitem__(key, value)
        else:
            raise ValueError("node key does not match number of layers")


class MultilayerGraph(nx.Graph):
    """
    Extent NetworkX nx.Graph class to handle multilayer networks.

    The underlying nx.Graph class is used to store the flattened network. The nodes in this network are the
    "state nodes" (i.e., "node-layer tuples") of the multilayer network.

    Any existing NetworkX methods should continue to work and simply operate on the flattened network.

    **Nodes:**

    Nodes in the multilayer network are tuples (n, a_1,...,a_k), where n is the node label, and a_i is the label for
    the ith aspect.

    node and aspect labels are stored in G.aspects (updated automatically when nodes are added)

    """
    def to_directed_class(self):
        def to_directed_copy():
            g = MultilayerDiGraph()
            g.aspects = deepcopy(self.aspects)
            return g
        return to_directed_copy

    def to_undirected_class(self):
        def to_undirected_copy():
            g = MultilayerGraph()
            g.aspects = deepcopy(self.aspects)
            return g
        return to_undirected_copy

    def __init__(self, incoming_graph_data=None, n_aspects=1, **attr):
        if hasattr(incoming_graph_data, 'aspects'):
            self.aspects = tuple(dict() for _ in incoming_graph_data.aspects)
        else:
            self.aspects = tuple(dict() for _ in range(n_aspects+1))
        self.adjlist_outer_dict_factory = lambda: adj_dict(self)
        self.node_dict_factory = lambda: node_dict(self)
        super().__init__(incoming_graph_data, **attr)

    def subgraph(self, nodes):
        sg = super().subgraph(nodes)
        sg.aspects = self.aspects
        return sg

    def edge_subgraph(self, edges):
        sg = super().edge_subgraph(edges)
        sg.aspects = self.aspects
        return sg

    def layer(self, l):
        """return specified layer of multilayer network as SubGraph view"""
        nodes = ((i, *l) for i in self.aspects[0])
        return self.subgraph(nodes)

    def interlayer(self, l1, l2=None):
        """return interlayer network for given layer (optional to other layer)"""
        nodes1 = ((i, *l1) for i in self.aspects[0])
        if l2 is not None:
            nodes2 = set((i, *l2) for i in self.aspects[0])
            edges = ((n1, n2) for n1 in nodes1 for n2 in self[n1] if n2 in nodes2)
        else:
            edges = ((n1, n2) for n1 in nodes1 for n2 in self[n1] if n2 not in nodes1)
        return self.edge_subgraph(edges)

    def make_fully_interconnected(self, **attr):
        for n in product(*self.aspects):
            if n not in self:
                self.add_node(n, **attr)

    def add_layer(self, layer, graph, node_mapping=None):
        """add layer from networkx graph

        layer: layer to add edges to
        graph: graph containing new edges
        node_mapping: map nodes of graph to nodes (not state-nodes) of multilayer network (optional if nodes are already
            consistent)
        """
        if node_mapping is None:
            def mapping(item):
                return (item, *layer)
        else:
            if hasattr(node_mapping, '__getitem__'):
                def mapping(item):
                    return (node_mapping[item], *layer)
            else:
                def mapping(item):
                    return (node_mapping(item), *layer)

        for node1, node2, data in graph.edges(data=True):
            self.add_edge(mapping(node1), mapping(node2))
            self[mapping(node1)][mapping(node2)].update(data)
            if not graph.is_directed() and self.is_directed():
                self.add_edge(mapping(node2), mapping(node1))
                self[mapping(node2)][mapping(node1)].update(data)

    def remove_layer(self, layer):
        """
        remove all nodes from a given layer of the network and remove layer from aspects
        :param layer:
        :return:
        """
        for i in self.aspects[0]:
            node = (i, *layer)
            if node in self:
                self.remove_node(node)
        for l, a in zip(layer, self.aspects[1:]):
            del(a[l])

    def add_categorical_coupling(self, aspect, self_coupling=False, **attr):
        """aspect > 0 as aspect[0] are the node labels"""
        for n1 in self.nodes:
            n2_list = list(n1)
            for a in self.aspects[aspect]:
                n2_list[aspect] = a
                n2 = tuple(n2_list)
                if n2 in self:
                    if self_coupling or n1 != n2:
                        self.add_edge(n1, n2, **attr)

    def remove_categorical_coupling(self, aspect, self_coupling=False):
        for n1 in self.nodes:
            n2_list = list(n1)
            for a in self.aspects[aspect]:
                n2_list[aspect] = a
                n2 = tuple(n2_list)
                if n2 in self:
                    if self_coupling or n1 != n2:
                        if self.has_edge(n1, n2):
                            self.remove_edge(n1, n2)

    def add_ordinal_coupling(self, aspect, key=None, reverse=False, **attr):
        """ aspect>0 and needs to be sortable. Optionally specify key for sorting"""
        aspect_keys = sorted(self.aspects[aspect], key=key, reverse=reverse)
        aspect_map = {k: i for i, k in enumerate(aspect_keys)}
        for n1 in self.nodes:
            n2_list = list(n1)
            for a in aspect_keys[aspect_map[n1[aspect]]+1:]:
                n2_list[aspect] = a
                n2 = tuple(n2_list)
                if n2 in self:
                    self.add_edge(n1, n2, **attr)
                    if self.is_directed():
                        self.add_edge(n2, n1, **attr) # make sure this behaves as expected in directed network
                    break

    def remove_ordinal_coupling(self, aspect, key=None, reverse=False):
        aspect_keys = sorted(self.aspects[aspect], key=key, reverse=reverse)
        aspect_map = {k: i for i, k in enumerate(aspect_keys)}
        for n1 in self.nodes:
            n2_list = list(n1)
            for a in aspect_keys[aspect_map[n1[aspect]]+1:]:
                n2_list[aspect] = a
                n2 = tuple(n2_list)
                if n2 in self:
                    if self.has_edge(n1, n2):
                        self.remove_edge(n1, n2)
                    if self.is_directed():
                        if self.has_edge(n2, n1):
                            self.remove_edge(n2, n1) # make sure this behaves as expected in directed network
                    break


class MultilayerDiGraph(MultilayerGraph, nx.DiGraph):
    def add_directed_ordinal_coupling(self, aspect, key=None, reverse=False, **attr):
        """ aspect>0 and needs to be sortable. Optionally specify key for sorting"""
        aspect_keys = sorted(self.aspects[aspect], key=key, reverse=reverse)
        aspect_map = {k: i for i, k in enumerate(aspect_keys)}
        for n1 in self.nodes:
            n2_list = list(n1)
            for a in aspect_keys[aspect_map[n1[aspect]]+1:]:
                n2_list[aspect] = a
                n2 = tuple(n2_list)
                if n2 in self:
                    self.add_edge(n1, n2, **attr)
                    break

    def remove_directed_ordinal_coupling(self, aspect, key=None, reverse=False):
        """ aspect>0 and needs to be sortable. Optionally specify key for sorting"""
        aspect_keys = sorted(self.aspects[aspect], key=key, reverse=reverse)
        aspect_map = {k: i for i, k in enumerate(aspect_keys)}
        for n1 in self.nodes:
            n2_list = list(n1)
            for a in aspect_keys[aspect_map[n1[aspect]]+1:]:
                n2_list[aspect] = a
                n2 = tuple(n2_list)
                if n2 in self:
                    self.remove_edge(n1, n2)
                    break

    def add_delay_ordinal_coupling(self, aspect, key=None, reverse=False, **attr):
        """keys for aspect should be numeric for reasonable results"""
        aspect_keys = sorted(self.aspects[aspect], key=key, reverse=reverse)
        aspect_map = {k: i for i, k in enumerate(aspect_keys)}
        for n1 in self.nodes:
            n2_list = list(n1)
            for a in aspect_keys[aspect_map[n1[aspect]]+1:]:
                n2_list[aspect] = a
                n2 = tuple(n2_list)
                if n2 in self:
                    self.add_edge(n1, n2, delay=n2[aspect]-n1[aspect], **attr)

    def remove_delay_ordinal_coupling(self, aspect, key=None, reverse=False):
        """keys for aspect should be numeric for reasonable results"""
        aspect_keys = sorted(self.aspects[aspect], key=key, reverse=reverse)
        aspect_map = {k: i for i, k in enumerate(aspect_keys)}
        for n1 in self.nodes:
            n2_list = list(n1)
            for a in aspect_keys[aspect_map[n1[aspect]] + 1:]:
                n2_list[aspect] = a
                n2 = tuple(n2_list)
                if n2 in self:
                    self.remove_edge(n1, n2)


def aggregate(mg, weighted=True):
    if mg.is_directed():
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    for n, data in mg.aspects[0].items():
        g.add_node(n)
        g.nodes[n].update(data)
    if weighted:
        for n1, n2, w in mg.edges(data='weight', default=1):
            if g.has_edge(n1[0], n2[0]):
                g[n1[0]][n2[0]]['weight'] += w
            else:
                g.add_edge(n1[0], n2[0], weight=w)
    else:
        for n1, n2 in mg.edges():
            g.add_edge(n1[0], n2[0])

    return g
