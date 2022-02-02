import itertools
import math
import time
import numpy as np
from matplotlib import ticker
from scipy import spatial
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

pi = np.pi

name_of_file = 'SampleCoordinates.txt'
radius = 0.08
start_node = 0
end_node = 5


def read_file(filename):
    with open(name_of_file, mode='r') as file:
        list_of_floats = []
        for line in file:
            strip_line = line[1:-1].strip("{}")
            replace_line = strip_line.replace(" ", "")
            split_line = replace_line.split(sep=',')

            for item in split_line:
                list_of_floats.append(float(item))

    list_to_array = np.array(list_of_floats)  # changing to array
    reshape_array = list_to_array.reshape(len(list_to_array) // 2, 2)  # reshape to a 2x2 array

    R = 1
    x = R * pi * reshape_array[:, 1] / 180
    y = R * np.log(np.tan((pi / 4 + pi * reshape_array[:, 0] / 360)))

    x_y = np.array(list(zip(x, y)))  # 2x2 array for the x and y coordinates, x-values column 1,  y-values column 2
    return (x_y)


coord_list = read_file(name_of_file)




def construct_graph_connections(coord_list, radius):
    distance_list = []
    first_index_list = []
    second_index_list = []
    for i, coord_i in enumerate(coord_list):
        for j in range(i + 1, len(coord_list)):
            coord_j = coord_list[j]
            d = coord_i - coord_j

            distance = math.sqrt(d[0] ** 2 + d[1] ** 2)

            if distance <= radius:
                distance_list.append(distance)
                first_index_list.append(i)
                second_index_list.append(j)
    index_array = np.array([first_index_list, second_index_list]).T
    distance_array = np.array(distance_list)

    return distance_array, index_array


start = time.time()
[distance, indices] = construct_graph_connections(coord_list, radius)
end = time.time()
print(end - start)

N = len(coord_list)
M = N


def construct_graph(indices, distance, N):
    sparse_graph = csr_matrix((distance, (indices[:, 0], indices[:, 1])), shape=(M, N))
    return sparse_graph


graph = construct_graph(indices, distance, N)


def find_shortest_path(graph, start_node, end_node):
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, indices=start_node,
                                              return_predecessors=True)
    path_length = dist_matrix[end_node]
    path = [end_node]
    current_node = end_node
    while current_node != start_node:
        current_node = predecessors[current_node]
        path.append(current_node)
    path = path[::-1]
    return path, path_length


[path, path_length] = find_shortest_path(graph, start_node, end_node)


def plot_points(coord_list, indices):
    fig, ax = plt.subplots()
    # ax.set_xlim(np.min(coord_list[:, 0]), np.max(coord_list[:, 0]))
    # ax.set_ylim(np.min(coord_list[:, 1]), np.max(coord_list[:, 1]))
    ax.axis('equal')
    lines = coord_list[indices]
    shortest_path = coord_list[path]
    ax.plot(coord_list[:, 0], coord_list[:, 1], '.', color='red', markersize=5, zorder=0)
    ax.plot(shortest_path[:, 0], shortest_path[:, 1], '-', color='blue', markersize=3)
    line_segments = LineCollection(lines, linewidths=0.3, colors='grey')
    ax.add_collection(line_segments)
    plt.title('Shortest path')
    ax.legend(['Cities', 'Shortest path'])
    plt.show()



def construct_fast_graph_connections(coord_list, radius):
    tree = cKDTree(coord_list)
    index_array = tree.query_pairs(radius, output_type='ndarray')
    print(index_array)

    distances = []
    for i, item in enumerate(index_array):
        first_col = index_array[i, 0]
        second_col = index_array[i, 1]
        dist = coord_list[first_col] - coord_list [second_col]
        distance = math.sqrt(dist[0]**2 + dist[1]**2)
        distances.append(distance)

    distance_array = np.array(distances)

    return distance_array, index_array


construct_fast_graph_connections(coord_list, radius)