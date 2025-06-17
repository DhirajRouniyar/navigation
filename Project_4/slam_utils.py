
import numpy as np
from pyqtgraph.Qt import QtWidgets
try:
    import PyQt5
    import pyqtgraph as pg
    import pyqtgraph.exporters
    can_plot = True
except ImportError:
    can_plot = False

def load_data(file_name):
    with open(file_name, "r") as f:
        raw_data = f.readlines()

    data = [ [float(x) for x in line.strip().split(',')] for line in raw_data ]

    return np.array(data)


def polar_to_cartesian(trees, ekf_state):
    if len(trees) == 0:
        return []

    trees = np.array(trees) # rows are [range, bearing, diameter]
    phi = ekf_state["x"][2]
    mu = ekf_state["x"][0:2]

    return np.reshape(mu, (2,1)) + np.vstack(( trees[:,0]*np.cos(phi+trees[:,1]), 
                                                trees[:,0]*np.sin(phi+trees[:,1])))


def draw_trees(points, matches, pose, plot):
    if len(points) == 0:
        return
    global_points = polar_to_cartesian(points, pose)
    robot_pos = pose["x"][:2]
    points_list = [pt.tolist() for pt in global_points.T]

    if "tree_lines" not in plot:
        plot["tree_lines"] = []
        plot["active_lines"] = []

    for i, match in enumerate(matches):
        segment = np.vstack((robot_pos, points_list[i]))
        color = 'b' if match >= 0 else ('g' if match == -2 else 'r')

        if i >= len(plot["tree_lines"]):
            line = plot["axis"].plot(segment, pen=pg.mkPen(color, width=2))
            plot["tree_lines"].append(line)
            plot["active_lines"].append(True)
        else:
            plot["tree_lines"][i].setData(segment, pen=pg.mkPen(color, width=2))
            if not plot["active_lines"][i]:
                plot["axis"].addItem(plot["tree_lines"][i])
                plot["active_lines"][i] = True

    for i in range(len(matches), len(plot["tree_lines"])):
        if plot["active_lines"][i]:
            plot["axis"].removeItem(plot["tree_lines"][i])
            plot["active_lines"][i] = False

def draw_path(path, plot):
    if path.size > 3:
        if "path" not in plot:
            plot["path"] = plot["axis"].plot()
        plot["path"].setData(path[:,:2], pen='k')

# def draw_landmarks(state, plot, config):
#     if "landmarks" not in plot:
#         plot["landmarks"] = plot["axis"].plot()
#     landmarks = state["x"][3:].reshape(-1, 2)
#     plot["landmarks"].setData(landmarks, pen=None, symbol="+", symbolPen='g', symbolSize=13)

#     if config.get("show_covariances", False):
#         if "cov_ellipses" not in plot:
#             plot["cov_ellipses"] = []
#         for i in range(state["num_landmarks"]):
#             idx = 3 + 2 * i
#             cov = state["P"][idx:idx+2, idx:idx+2]
#             ellipse = get_ellipse(state["x"][idx:idx+2], cov)
#             if i >= len(plot["cov_ellipses"]):
#                 plot["cov_ellipses"].append(plot["axis"].plot())
#             plot["cov_ellipses"][i].setData(ellipse, pen='b')

def draw_landmarks(ekf_state, plot, params):
    if "map" not in plot:
        plot["map"] = plot["axis"].plot()

    lms = np.reshape(ekf_state["x"][3:], (-1, 2))
    plot["map"].setData(lms, pen=None, symbol="+", symbolPen='g', symbolSize=13)

    if params["plot_map_covariances"]:
        if "map_covariances" not in plot:
            plot["map_covariances"] = []

        for i in range(ekf_state["num_landmarks"]):
            idx = 3 + 2*i
            P = ekf_state["P"][idx:idx+2, idx:idx+2]

            circ = get_ellipse(ekf_state["x"][idx:idx+2], P)

            if i >= len(plot["map_covariances"]):
                plot["map_covariances"].append(plot["axis"].plot())

            plot["map_covariances"][i].setData(circ, pen='b')

def get_ellipse(center, cov, base=[]):
    if not base:
        theta = np.linspace(0, 2*np.pi, 20)
        circle = np.hstack((np.cos(theta).reshape(-1,1), np.sin(theta).reshape(-1,1)))
        base.extend(circle.tolist())
    vals, _ = np.linalg.eigh(cov)
    offset = 1e-6 - min(0, vals.min())
    L = np.linalg.cholesky(cov + offset * np.eye(center.shape[0]))
    return 3 * np.dot(np.array(base), L.T) + center

def scan_to_xy(pose, scan, config):
    angles = np.linspace(-np.pi/2, np.pi/2, 361)
    valid = scan < config["max_range"]
    scan = scan[valid]
    angles = angles[valid]

    x = scan * np.cos(pose["x"][2] + angles)
    y = scan * np.sin(pose["x"][2] + angles)
    return pose["x"][:2].reshape(2,1) + np.vstack((x, y))

def draw_scan(pose, scan, plot, config):
    if "scan" not in plot:
        plot["scan"] = plot["axis"].plot()
    xy = scan_to_xy(pose, scan, config)
    plot["scan"].setData(xy.T, pen=None, symbol="d", symbolPen='k', symbolSize=3)

def draw_robot(pose, plot):
    triangle = 1.5 * np.array([[0, 0], [-3, 1], [-3, -1], [0, 0]])
    angle = pose["x"][2]
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    triangle = triangle @ R.T + pose["x"][:2]
    if "robot" not in plot:
        plot["robot"] = plot["axis"].plot()
    plot["robot"].setData(triangle, pen=pg.mkPen('k', width=2))

def draw_pose_cov(pose, plot):
    ellipse = get_ellipse(pose["x"][:2], pose["P"][:2, :2])
    if "pose_cov" not in plot:
        plot["pose_cov"] = plot["axis"].plot()
    plot["pose_cov"].setData(ellipse, pen='b')

def show_state(pose, plot, config):
    draw_landmarks(pose, plot, config)
    draw_robot(pose, plot)
    draw_pose_cov(pose, plot)

def update_display(path, pose, trees, scan, matches, plot, config, count):
    draw_path(path, plot)
    show_state(pose, plot, config)
    draw_trees(trees, matches, pose, plot)
    if len(scan) > 0 and config.get("show_scan", False):
        draw_scan(pose, scan, plot, config)
    QtWidgets.QApplication.processEvents()

def setup_plot():
    if not can_plot:
        raise Exception("Missing PyQt5 or pyqtgraph")
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    win = pg.GraphicsLayoutWidget()
    win.setWindowTitle("SLAM Viewer")
    win.show()
    canvas = win.addPlot()
    canvas.setAspectLocked(True)
    return {"win": win, "axis": canvas}

def wrap_angle(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle


def symmetrize(mat):
    return 0.5 * (mat + mat.T)


def greedy_match(cost_matrix):
    num = cost_matrix.shape[0]
    results = []
    order = np.argsort(cost_matrix.min(axis=1))
    for i in order:
        j = np.argmin(cost_matrix[i])
        cost_matrix[:, j] = 1e8
        results.append((i, j))
    return results