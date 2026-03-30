import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict

from networkx.algorithms.bipartite.cluster import clustering


def create_graph(edges):  # edges a list [(i, j, w)], with w the weight on the (i, j) edge
    g = nx.Graph()
    g.add_weighted_edges_from(edges)
    return g

def mutex_watershed(graph, connect_all):
    V = graph[0]
    weight_dict = graph[1]

    # here E is sorted according to absolute value of w in graph
    E = {k: v for k, v in sorted(weight_dict.items(), key=lambda item: abs(item[1]), reverse=True)}
    a_pos = UnionFind(False)
    a_neg = UnionFind(True)
    c = Clustering(a_pos, a_neg)

    #TODO USE CLUSTERING STRUCT
    for e in E:
        i = e[0]
        j = e[1]
        if weight_dict[e] > 0:
            if not a_neg.is_mutex(i, j):
                if not a_pos.connected(i,j) or not connect_all:
                # if not connected(e, a_pos) or not connect_all:
                    c.merge(i,j)

        if not a_pos.connected(i,j):
            # a_neg.merge(i,j)  # add_mutex(e, a_neg)
            # a_neg.add_mutex(i,j)
            c.split(i,j)

    # print("HERE")
    # print(a_pos.parent)
    # print()
    # print(a_neg.parent)
    # print(a_neg.mutexes)
    # print()
    # print(c)

    return c


class UnionFind:
    def __init__(self, is_mutex_enabled=False):
        self.parent = {}  # Maps character to its parent
        self.rank = {}  # Maps character to its rank
        self.mutexes = {}  # Maps character to a set of mutexes
        self.is_mutex_enabled = is_mutex_enabled  # Flag to enable or disable mutex functionality

    def _ensure(self, x):
        """Ensures that a character is in the parent, rank, and mutex set dictionaries."""
        if x not in self.parent:
            self.parent[x] = x  # Node is its own parent
            self.rank[x] = 0  # Rank is 0
            self.mutexes[x] = set() if self.is_mutex_enabled else None  # No mutexes if disabled

    def find(self, x):
        """Find the root of the character `x`."""
        self._ensure(x)
        root = self.parent[x]
        if self.parent[root] != root:
            self.parent[x] = self.find(root)
            return self.parent[x]
        return root
        # if self.parent[x] != x:
        #     self.parent[x] = self.find(self.parent[x])  # Path compression
        # return self.parent[x]

    def merge(self, x, y):
        """Union the sets containing `a` and `b`."""
        self._ensure(x)
        self._ensure(y)
        xRoot = self.find(x)
        yRoot = self.find(y)
        if xRoot == yRoot:  # already merged
            return
        # Union by Rank
        if self.rank[xRoot] < self.rank[yRoot]:
            self.parent[xRoot] = yRoot
        elif self.rank[yRoot] < self.rank[xRoot]:
            self.parent[yRoot] = xRoot
        else:
            self.parent[yRoot] = xRoot
            self.rank[xRoot] += 1
            # Merge the mutex sets if enabled
        if self.is_mutex_enabled:
            # self.add_mutex(x, y)
            self.mutexes[xRoot].update(self.mutexes[yRoot])
            self.mutexes[yRoot].update(self.mutexes[xRoot])
        # rootA = self.find(a)
        # rootB = self.find(b)
        #
        # if rootA != rootB:
        #     # Perform the union by rank
        #     if self.rank[rootA] > self.rank[rootB]:
        #         self.parent[rootB] = rootA
        #     elif self.rank[rootA] < self.rank[rootB]:
        #         self.parent[rootA] = rootB
        #     else:
        #         self.parent[rootB] = rootA
        #         self.rank[rootA] += 1
        #
        #     # Merge the mutex sets if enabled
        #     if self.is_mutex_enabled:
        #         self.mutexes[rootA].update(self.mutexes[rootB])
        #         self.mutexes[rootB].update(self.mutexes[rootA])

    def connected(self, i, j):
        return self.find(i) == self.find(j)

    def add_mutex(self, a, b):
        """Add a mutex relationship between characters `a` and `b`."""
        if not self.is_mutex_enabled:
            return  # Ignore if mutex is not enabled

        self._ensure(a)
        self._ensure(b)

        rootA = self.find(a)
        rootB = self.find(b)

        if rootA != rootB:
            self.mutexes[rootA].add(b)
            self.mutexes[rootB].add(a)

    def is_mutex(self, a, b):
        """Check if `a` and `b` are mutexes (incompatible)."""
        if not self.is_mutex_enabled:
            return False  # Mutex checking is disabled

        self._ensure(a)
        self._ensure(b)

        rootA = self.find(a)
        rootB = self.find(b)

        # If they are in the same group and one is in the other's mutex set
        return b in self.mutexes[rootA] or a in self.mutexes[rootB]

    def get_mutexes(self, x):
        """Return the set of mutexes for a character."""
        if not self.is_mutex_enabled:
            return set()  # No mutexes if not enabled
        self._ensure(x)
        root = self.find(x)
        return self.mutexes[root]

#
# # copied from geeksforgeeks then modified
# class DisjointUnionSets:
#     def __init__(self):
#         self.parent = {}
#         self.rank = {}
#         # self.rank = [0] * n
#         # self.parent = list(range(n))
#     #
#     def add_to_set(self, v):
#         self.parent[v] = v
#         self.rank[v] = 0
#
#     def find_root(self, i):
#         root = self.parent[i]
#         if self.parent[root] != root:
#             self.parent[i] = self.find_root(root)
#             return self.parent[i]
#         return root
#
#     def connected(self, i, j):
#         return self.find_root(i) == self.find_root(j)
#
#     def merge(self, x, y):
#         xRoot = self.find_root(x)
#         yRoot = self.find_root(y)
#         if xRoot == yRoot:  #already merged
#             return
#         # Union by Rank
#         if self.rank[xRoot] < self.rank[yRoot]:
#             self.parent[xRoot] = yRoot
#         elif self.rank[yRoot] < self.rank[xRoot]:
#             self.parent[yRoot] = xRoot
#         else:
#             self.parent[yRoot] = xRoot
#             self.rank[xRoot] += 1

class Clustering:
    def __init__(self, a_pos, a_neg):
        self.positives = a_pos  # UnionFind(False)  # UnionFind for positive relationships
        self.negatives = a_neg  # UnionFind(True)  # UnionFind for negative relationships

    def merge(self, a, b):
        self.positives.merge(a, b)
        root = self.positives.find(a)
        if root == a:
            self.negatives.merge(a, b)  # todo check that this a b vs b a is not impacted by the initial merge func?
        else:
            self.negatives.merge(b, a)

    def split(self, a, b):
        self.negatives.add_mutex(a, b)

    def clusters(self):
        cluster = {}
        for k, v in self.positives.parent.items():
            if v not in cluster.keys():
                cluster[v] = [k]
            else:
                cluster[v] += [k]
        return cluster

    def __repr__(self):

        return str(self.clusters())



if __name__ == '__main__':
    g = [('a', 'b', 'c', 'd', 'e'),
         {('a', 'b'): 1, ('b', 'c'): 3, ('b', 'd'):-7, ('c', 'd'):2, ('d', 'e'):5, ('e', 'a'):6, ('e', 'b'):4}]
    # {('a', 'b'): 1, ('b', 'c'): 3, ('b', 'd'): -7, ('c', 'd'): 2, ('d', 'e'): 5, ('e', 'a'): 6, ('e', 'b'): 4}]

    # g = [('a', 'b', 'c', 'd', 'e'),
         # {('a', 'b'): 1, ('b', 'c'): 3, ('b', 'd'): -7, ('c', 'd'): 2, ('d', 'e'): 5, ('e', 'a'): -4}]
    # g = [('a', 'b'), {('a', 'b'): -3}]

    clus = mutex_watershed(g, False)
    print(clus)
    # d = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    # dd= d_sorted_by_value = OrderedDict(sorted(d.items(), key=lambda x: x[1]))
    # print(dd)
    # x = {(1, 3): 2, (3, 8): 4, (9, 2): -3, (7, 1): -1, (3, 6): 0}
    # m = {k: v for k, v in sorted(x.items(), key=lambda item: abs(item[1]), reverse=True)}
    # print(m)

    # edge = [('a', 'b', 2), ('a', 'c', 4), ('b', 'd', 1), ('d', 'e', 5), ('c', 'b', 3)]
    # g = create_graph(edge)
    # print(g.get_edge_data('a', 'd'))
    # nx.draw(g)
    # plt.show()