import math
import matplotlib.pyplot as plt
import numpy as np
import random
import heapq
from matplotlib.patches import Rectangle

def load_map(file_name):
    grid = []
    with open(file_name, 'r') as file:
        for line in file:
            # Strip the newline character and split by spaces
            grid.append(line.strip().split())
    
    # Convert the grid to a 2D list of bools
    for i in range(len(grid)):
        string = grid[i][0]
        row = []
        for j in range(len(string)):
            if string[j] == '.':
                row.append(False)
            elif string[j] == 'X':
                row.append(True)
        grid[i] = row
            
    return grid

def compute_path_distance(route):
    total_dist = 0
    for idx in range(1, len(route)):
        total_dist += math.sqrt(sum((route[idx][dim] - route[idx-1][dim])**2 for dim in range(len(route[idx]))))
    return total_dist

def fetch_neighbors(map_data, position):
    i, j = position
    movement_dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    ]
    neighbors = []

    for dx, dy in movement_dirs:
        ni, nj = i + dx, j + dy
        if 0 <= ni < len(map_data) and 0 <= nj < len(map_data[0]):
            if not map_data[ni][nj]:
                if abs(dx) + abs(dy) == 2 and (map_data[i][nj] or map_data[ni][j]):
                    continue
                neighbors.append((ni, nj))
    return neighbors

def bfs_search(map_data, origin, destination):
    if map_data[destination[0]][destination[1]] or map_data[origin[0]][origin[1]]:
        return None
    explored = [[False] * len(map_data[0]) for _ in range(len(map_data))]
    previous = [[0] * len(map_data[0]) for _ in range(len(map_data))]
    q = []
    q.append(origin)
    explored[origin[0]][origin[1]] = True
    previous[origin[0]][origin[1]] = -1

    while(len(q) != 0):
        current = q.pop(0)
        for nbr in fetch_neighbors(map_data, current):
            if not explored[nbr[0]][nbr[1]]:
                q.append(nbr)
                explored[nbr[0]][nbr[1]] = True
                previous[nbr[0]][nbr[1]] = current
                if nbr[0] == destination[0] and nbr[1] == destination[1]:
                
                    trace = []
                    while previous[nbr[0]][nbr[1]] != -1:
                        trace.append(nbr)
                        nbr = previous[nbr[0]][nbr[1]]
                    trace.append(origin)
                    trace.reverse()
                    return trace
    return None



def rrt_explore(map_data, origin, destination, max_iters=5000):
    if map_data[destination[0]][destination[1]] or map_data[origin[0]][origin[1]]:
        return None

    def is_valid(point):
        x, y = point
        return 0 <= x < len(map_data) and 0 <= y < len(map_data[0]) and not map_data[x][y]

    def euclidean(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def nearest_direction(parent, random_target):
        neighbors = fetch_neighbors(map_data, parent)
        if not neighbors:
            return None
        return min(neighbors, key=lambda node: euclidean(node, random_target))

    path_tree = {tuple(origin): None}
    all_nodes = [tuple(origin)]

    for attempt in range(max_iters):
        print("Iteration:", attempt, end="\r")
        sampled = (random.randint(0, len(map_data)-1), random.randint(0, len(map_data[0])-1))
        closest = min(all_nodes, key=lambda node: euclidean(node, sampled))
        extension = nearest_direction(closest, sampled)

        if extension and extension not in path_tree:
            path_tree[extension] = closest
            all_nodes.append(extension)

            if euclidean(extension, destination) <= 1.5 and is_valid(destination):
                if tuple(destination) not in path_tree:
                    path_tree[tuple(destination)] = extension
                    all_nodes.append(destination)
                break

    if tuple(destination) in path_tree:
        trace = []
        node = tuple(destination)
        while node is not None:
            trace.append(node)
            node = path_tree[node]
        trace.reverse()
        return trace

    return None

def dijkstra_search(map_data, origin, destination):
    if map_data[destination[0]][destination[1]] or map_data[origin[0]][origin[1]]:
        return None

    visited = [[False] * len(map_data[0]) for _ in range(len(map_data))]
    previous = [[False] * len(map_data[0]) for _ in range(len(map_data))]
    cost = [[float('inf')] * len(map_data[0]) for _ in range(len(map_data))]
    pq = []
    
    heapq.heappush(pq, (0, origin))
    previous[origin[0]][origin[1]] = -1
    cost[origin[0]][origin[1]] = 0

    # while pq:
    while(len(pq) > 0):
        current_cost, current_node = heapq.heappop(pq)
        visited[current_node[0]][current_node[1]] = True
        neighbours = fetch_neighbors(map_data, current_node)

        for nbr in neighbours:
            if not visited[nbr[0]][nbr[1]]:
                tentative_cost = 1 + cost[current_node[0]][current_node[1]]

                if tentative_cost < cost[nbr[0]][nbr[1]]:
                    cost[nbr[0]][nbr[1]] = tentative_cost
                    previous[nbr[0]][nbr[1]] = current_node
                    heapq.heappush(pq, (tentative_cost, nbr))

                if nbr[0] == destination[0] and nbr[1] == destination[1]:
                    trace = []
                    trace.append(destination)
                    node = destination
                    while previous[node[0]][node[1]] != -1:
                        trace.append(previous[node[0]][node[1]])
                        node = previous[node[0]][node[1]]
                    trace.append(origin)
                    trace.reverse()
                    return trace

    return None

def render_map(map_data, route=None, origin=None, destination=None):
    colormap = plt.cm.get_cmap('Greys')
    colormap.set_under(color='white')
    colormap.set_over(color='black')

    grid_np = np.asarray(map_data)
    fig, ax = plt.subplots()

    ax.matshow(grid_np, cmap=colormap, vmin=0.1, vmax=1.0)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(-0.5, len(map_data[0]), 1))
    ax.set_yticks(np.arange(-0.5, len(map_data), 1))
    ax.set_xticklabels(range(0, len(map_data[0])+1))
    ax.set_yticklabels(range(0, len(map_data)+1))

    # Highlight route cells with a blue overlay
    if route:
        for y, x in route:
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1, color='blue', alpha=0.5)
            ax.add_patch(rect)

    # Mark origin (green)
    if origin:
        ax.plot(origin[1], origin[0], 'go')

    # Mark destination (red)
    if destination:
        ax.plot(destination[1], destination[0], 'ro')

    return fig

