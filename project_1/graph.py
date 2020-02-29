# _*_coding:utf-8 _*_
#author: Yibo Fu
#G25190736

from collections import namedtuple
from pathlib import Path
from queue import PriorityQueue

from tools import timeit


Node = namedtuple("Node", ["id", "square", "x", "y"])
Edge = namedtuple("Edge", ["from_", "to_", "dist"])


def parse_node(line):
    v, square = line.split(",")
    square = int(square)
    x, y = square // 10, square % 10
    return Node(int(v), square, x, y)


def parse_edge(line):
    from_vertex, to_vertex, distance = line.split(",")
    return Edge(int(from_vertex), int(to_vertex), float(distance))


def create_graph_from_folder(folder):
    graph = Graph()
    folder = Path(folder)
    with open(folder / "v.txt") as v:
        for line in v:
            if line.startswith("#"):
                continue
            node = parse_node(line)
            graph.add_node(node)
    with open(folder / "e.txt") as e:
        for line in e:
            if line.startswith("#"):
                continue
            edge = parse_edge(line)
            graph.add_edge(edge)
    return graph


def parse_path(start, end, shortest):
    if end not in shortest:
        return None
    path = []
    node = end
    while node != -1:
        dist, prev = shortest[node]
        path.append((node, dist))
        node = prev
    path.reverse()
    return path


@timeit
def dijkstra(start, end, graph):
    shortest = {}
    cur = PriorityQueue(len(graph.nodes) ** 2)
    cur.put((0.0, (start, -1)))
    i = 0
    while not cur.empty():
        dist, (node, lastNode) = cur.get()
        if node in shortest:
            continue
        shortest[node] = (dist, lastNode)
        if node == end:
            break
        for _, neighbour, d in graph.edges[node]:
            if neighbour in shortest:
                continue
            cur.put((dist + d, (neighbour, node)))
            i += 1
    print(f"Iteration for {i} times")
    return parse_path(start, end, shortest)


def square_to_pos(square):
    return square // 10, square % 10


@timeit
def a_star(start, end, graph):
    def heuristic(lhs, rhs):
        lhs = graph.get_node(lhs)
        rhs = graph.get_node(rhs)
        # use Manhattan Distance
        return abs(lhs.x - rhs.x) + abs(lhs.y - rhs.y)
    shortest = {}
    open_list = PriorityQueue(len(graph.nodes) ** 2)
    open_list.put((0.0, (start, -1, 0)))
    i = 0
    while not open_list.empty():
        f, (node, last_node, path_length) = open_list.get()
        if node in shortest and path_length >= shortest[node][0]:
            continue
        shortest[node] = (path_length, last_node)
        if node == end:
            break
        for _, neighbour, d in graph.edges[node]:
            if neighbour in shortest:
                continue
            g = path_length + d
            f = g + heuristic(neighbour, end)
            open_list.put((f, (neighbour, node, g)))
            i += 1
    print(f"Iteration for {i} times")
    return parse_path(start, end, shortest)

class Graph:
    def __init__(self):
        self._nodes = {}
        self._edges = {}

    def get_node(self, node_id):
        return self._nodes[node_id]

    @property
    def nodes(self):
        return self._nodes.keys()

    @property
    def edges(self):
        return self._edges

    def add_node(self, node: Node):
        self._nodes[node.id] = node
        self._edges[node.id] = []

    def add_edge(self, edge: Edge):
        self._edges[edge.from_].append(edge)
        self._edges[edge.to_].append(Edge(edge.to_, edge.from_, edge.dist))
