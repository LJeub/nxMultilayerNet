"""
Work with multilayer networks on top of NetworkX
"""
import networkx as nx
from copy import deepcopy
from itertools import product
from collections.abc import Iterable
from pkg_resources import get_distribution, DistributionNotFound


# define __version__
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


class NodeDict(dict):
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
            raise ValueError("Node key does not match number of layers")


class AdjDict(dict):
    """Helper class to ensure all new nodes have correct format"""
    def __init__(self, mg):
        super().__init__()
        self._mg = mg

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == len(self._mg.aspects):
            super().__setitem__(key, value)
        else:
            raise ValueError("Node key does not match number of layers")


class Layers:
    """
    Implement layer slicing.
    """
    def __init__(self, mg):
        self._mg = mg

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.start is None and item.stop is None and item.step is None:
                layers = product(*self._mg.aspects[1:])
            else:
                if len(self._mg.aspects) == 2:
                    layers = range(*item.indices((len(self._mg.aspects[1]))))
                else:
                    raise ValueError("wrong number of dimensions for layer indexing")
        elif isinstance(item, tuple):
            layer_iter = []
            for li, ai in zip(item, self._mg.aspects[1:]):
                if isinstance(li, slice):
                    if li.start is None and li.stop is None and li.step is None:
                        layer_iter.append(ai)
                    else:
                        layer_iter.append(range(*li.indices(len(ai))))
                elif isinstance(li, Iterable):
                    layer_iter.append(li)
                else:
                    layer_iter.append((li,))
            layers = product(*layer_iter)
        else:
            if len(self._mg.aspects) == 2:
                layers = [(item,)]
            else:
                raise ValueError("wrong number of dimensions for layer indexing")
        edges = []
        for l in layers:
            nodes = set((i, *l) for i in self._mg.aspects[0])
            nodes.intersection_update(self._mg.nodes)
            edges.extend((n1, n2) for n1 in nodes for n2 in self._mg.adj[n1] if n2 in nodes)
        return self._mg.edge_subgraph(edges)


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
            self.aspects = deepcopy(incoming_graph_data.aspects)
        else:
            self.aspects = tuple(dict() for _ in range(n_aspects+1))
        self.layers = Layers(self)
        self.adjlist_outer_dict_factory = lambda: AdjDict(self)
        self.node_dict_factory = lambda: NodeDict(self)
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
        return self.layers[l]

    def interlayer(self, l1, l2=None):
        """return interlayer network for given layer (optional to other layer)"""
        nodes1 = self.layers[l1].nodes
        if l2 is not None:
            nodes2 = self.layers[l2].nodes
            edges = ((n1, n2) for n1 in nodes1 for n2 in self[n1] if n2 in nodes2)
        else:
            edges = ((n1, n2) for n1 in nodes1 for n2 in self[n1] if n2 not in nodes1)
        return self.edge_subgraph(edges)

    def interlayer_edges(self, **kwargs):
        return (e for e in self.edges(**kwargs) if e[0][1:] != e[1][1:])

    def intralayer_edges(self, **kwargs):
        return (e for e in self.edges(**kwargs) if e[0][1:] == e[1][1:])

    def make_fully_interconnected(self, **attr):
        """add any missing state nodes to the network such that each physical node is represented in each layer."""
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
        if not isinstance(layer, Iterable):
            layer = (layer,)

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
        if not isinstance(layer, Iterable):
            layer = (layer,)

        for i in self.aspects[0]:
            node = (i, *layer)
            if node in self:
                self.remove_node(node)
        for l, a in zip(layer, self.aspects[1:]):
            del(a[l])

    def remove_physical_node(self, node):
        """Remove all state nodes representing a particular physical node and remove the physical node from aspects."""
        if node in self.aspects[0]:
            for sn in product((node,), *self.aspects[1:]):
                if sn in self:
                    self.remove_node(sn)
            del(self.aspects[0][node])

    def add_categorical_coupling(self, aspect=1, self_coupling=False, **attr):
        """aspect > 0 as aspect[0] are the node labels"""
        for n1 in self.nodes:
            n2_list = list(n1)
            for a in self.aspects[aspect]:
                n2_list[aspect] = a
                n2 = tuple(n2_list)
                if n2 in self:
                    if self_coupling or n1 != n2:
                        self.add_edge(n1, n2, **attr)

    def remove_categorical_coupling(self, aspect=1, self_coupling=False):
        for n1 in self.nodes:
            n2_list = list(n1)
            for a in self.aspects[aspect]:
                n2_list[aspect] = a
                n2 = tuple(n2_list)
                if n2 in self:
                    if self_coupling or n1 != n2:
                        if self.has_edge(n1, n2):
                            self.remove_edge(n1, n2)

    def add_ordinal_coupling(self, aspect=1, key=None, reverse=False, **attr):
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

    def remove_ordinal_coupling(self, aspect=1, key=None, reverse=False):
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
    def add_directed_ordinal_coupling(self, aspect=1, key=None, reverse=False, **attr):
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

    def remove_directed_ordinal_coupling(self, aspect=1, key=None, reverse=False):
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

    def add_delay_ordinal_coupling(self, aspect=1, key=None, reverse=False, **attr):
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

    def remove_delay_ordinal_coupling(self, aspect=1, key=None, reverse=False):
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


def aggregate(mg, weighted=True, weight='weight', default=1, aggregate_weight='weight'):
    """
    Create aggregate network from multilayer network and return as nx.Graph or nx.DiGraph

    :param mg: multilayer network
    :param weighted: bool (default: True) select whether aggregate network should be weighted (default) or not
    :param weight: attribute name for storing weights (default: 'weight')
    :param default: default value for edges without weight (default: 1)
    :param aggregate_weight: (default: 'weight') attribute name for storing weights in aggregate network
    :return: aggregate network: (nx.DiGraph if mg.is_directed(): else: nx.Graph)
    """

    if mg.is_directed():
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    for n, data in mg.aspects[0].items():
        g.add_node(n)
        g.nodes[n].update(data)
    if weighted:
        for n1, n2, w in mg.edges(data=weight, default=default):
            if g.has_edge(n1[0], n2[0]):
                g[n1[0]][n2[0]][aggregate_weight] += w
            else:
                g.add_edge(n1[0], n2[0])
                g[n1[0]][n2[0]][aggregate_weight] = w
    else:
        for n1, n2 in mg.edges():
            g.add_edge(n1[0], n2[0])

    return g
def write_gml(mg, path, stringizer=nx.readwrite.gml.literal_stringizer):
    """
    Write multilayer graph in GML format.

    TODO: Implement storing of aspect information

    :param mg: multilayer graph
    :param path: path
    :param stringizer: (optional) function to convert python values to strings (needs to at least support writing tuples)
    """
    nx.readwrite.write_gml(mg, path, stringizer)


def read_gml(path, label='label', destringizer=nx.readwrite.gml.literal_destringizer):
    """
    Read multilayer graph from GML formatted file

    TODO: Implement storing of aspect information

    :param path: Path to GML file
    :param destringizer: (optional) function to convert python values from strings (needs to at least support reading
                       tuples)
    :return: MultilayerGraph or MultilayerDiGraph
    """
    g = nx.readwrite.read_gml(path, label, destringizer)
    nodes = list(g.nodes)
    if nodes:
        if g.is_directed():
            mg = MultilayerDiGraph(g, n_aspects=len(nodes[0])-1)
        else:
            mg = MultilayerGraph(g, n_aspects=len(nodes[0])-1)
    else:
        mg = MultilayerGraph()
    return mg
