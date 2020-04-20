import random


import networkx as nx
import matplotlib.pyplot as plt

from numpy import round

from .node import InternalNode

"""
the foolowing code was taken from: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3
much thanks to the author
"""


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


class NodeBfsWrapper:
    def __init__(self, node, name):
        self.node_data = node
        self.name = name


class TreeVisualizer:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def number_nodes(self, root):
        """number the bodes with bfs algorithm"""
        root = NodeBfsWrapper(root, 0)
        queue = [[root]]
        counter = 0
        while queue:
            level_nodes = queue.pop(0)
            next_level_nodes = []
            for node in level_nodes:
                self.nodes.append(node)
                if isinstance(node.node_data, InternalNode):
                    left_child = NodeBfsWrapper(node.node_data.left, counter +1)
                    right_child = NodeBfsWrapper(node.node_data.right, counter +2)
                    self.edges.append((node.name, left_child.name))
                    self.edges.append((node.name, right_child.name))
                    next_level_nodes.append(left_child)
                    next_level_nodes.append(right_child)
                    counter += 2
            if next_level_nodes:
                queue.append(next_level_nodes)

    def plot(self, root):
        self.number_nodes(root)
        g = nx.Graph()
        g.add_edges_from(self.edges)
        # TODO: add more data to node rather than it's name
        labeldict = {}
        for node in self.nodes:
            field = node.node_data.field if isinstance(node.node_data,InternalNode) else 'leaf'
            purity = round(node.node_data.purity,2) #  if isinstance(node.node_data,InternalNode) else ''
            labeldict[node.name] = F"{node.name}_{field}\n{purity} \n{node.node_data.n_examples}"
        pos = hierarchy_pos(g, 0)
        nx.draw(g, pos=pos, labels=labeldict,with_labels=True)
        plt.show()

