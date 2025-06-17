import matplotlib.pyplot as plt
import numpy as np
import utils
import json
from time import time
import tracemalloc

def run_algorithm(algorithm_fn, algorithm_name, grid, start, goal, key, max_iters=None):
    print(f"\n========== {algorithm_name.upper()} ==========")
    start_time = time()
    tracemalloc.start()

    if max_iters:
        path = algorithm_fn(grid, start, goal, max_iters=max_iters)
    else:
        path = algorithm_fn(grid, start, goal)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time()

    print(f"Start: {start}, Goal: {goal}")
    print(f"Memory usage {algorithm_name}: {current / 1e6:.4f} MB; Peak: {peak / 1e6:.4f} MB")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    if path is not None:
        path_length = utils.compute_path_distance(path)
     
        print(f"Path length: {path_length}")
    else:
        print("No path found!")

    fig = utils.render_map(grid, path, start, goal)
  
    plt.savefig(f"{algorithm_name.lower()}_grid_3_startgoalpair_{key}.png")
    plt.close(fig)
    print("="*40)

if __name__ == "__main__":
    grid_path = "D:\WPI\Advanced Robot Nav\Assignments\Assignment 1\map2.txt"
    start_goal_path = "start_goal_2.json"

    grid = utils.load_map(grid_path)

    with open(start_goal_path, 'r') as file:
        start_goal = json.load(file)

    for key, (start, goal) in start_goal.items():
        run_algorithm(utils.bfs_search, "BFS", grid, start, goal, key)
        run_algorithm(utils.dijkstra_search, "Dijkstra", grid, start, goal, key)
        run_algorithm(utils.rrt_explore, "RRT", grid, start, goal, key, max_iters=5000)

