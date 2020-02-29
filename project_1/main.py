# _*_coding:utf-8 _*_
#author: Yibo Fu
#G25190736

import argparse
import random
from graph import create_graph_from_folder, dijkstra, a_star

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(
        prog="Shortest Path",
        description="Find the shortest path from start to end in a graph"
    )
    parser.add_argument(
        "graph_folder",
        help="location of graph folder"
    )
    parser.add_argument(
        "-i", "--informed",
        type=bool, default=False, dest="is_informed",
        help="whether use informed search algorithm"
    )
    parser.add_argument(
        "--start",
        type=int, default=None, dest="start",
        help="start point of the path"
    )
    parser.add_argument(
        "--end",
        type=int, default=None, dest="end",
        help="end point of the path"
    )
    args = parser.parse_args()

    # read graph file and construct the graph
    graph = create_graph_from_folder(args.graph_folder)

    # get the start and end points
    start, end = random.sample(graph.nodes, 2)
    start = args.start if args.start is not None else start
    end = args.end if args.end is not None else end

    # search the shortest path from start to end
    # if args.is_informed:
    #     path = a_star(start, end, graph)
    # else:
    #     path = dijkstra(start, end, graph)

    path_1 = a_star(start, end, graph)
    print(f"Shortest path from {start} to {end}:")
    print(path_1)

    print(' ')

    path_2 = dijkstra(start, end, graph)
    print(f"Shortest path from {start} to {end}:")
    print(path_2)
