import matplotlib.pyplot as plt
import numpy as np
import observation_model as obs


def simulation(filename):
    """
    Plot the estimated position and orientation,
    compare with the ground truth data
    Inputs:
       filename: file path
    Output:
       plots
    """
    pos = []
    ort = []
    est_t = []
    model = obs.Observation_Model()
    data, ground_truth, ground_truth_t = model.read_data(filename)

    for index, data_ in enumerate(data):
        if isinstance(data_['id'], int):
            p, r = model.estimate_pose([data_['id']], index)
            pos.append(p)
            ort.append(r)
            est_t.append(float(data_['t']))
            continue
        if data_['id'].shape[0] > 1:
            p, r = model.estimate_pose(data_['id'], index)
            pos.append(p)
            ort.append(r)
            est_t.append(float(data_['t']))

    estimated = np.zeros((6, len(pos)))
    estimated[0] = np.array([p[0] for p in pos])
    estimated[1] = np.array([p[1] for p in pos])
    estimated[2] = np.array([p[2] for p in pos])
    estimated[3] = np.array([r[0] for r in ort])
    estimated[4] = np.array([r[1] for r in ort])
    estimated[5] = np.array([r[2] for r in ort])

    plot_fig = plt.figure()
    plot_graph = plot_fig.add_subplot(111, projection='3d')

    # Plot
    plot_graph.plot(ground_truth[0], ground_truth[1], ground_truth[2], c='g', label='Ground Truth')
    plot_graph.plot(estimated[0], estimated[1], estimated[2], 'r--', label='Estimated')
    plot_graph.legend()
    plot_graph.set_xlabel('X_axis')
    plot_graph.set_ylabel('Y_axis')
    plot_graph.set_zlabel('Z_Axis')
    plot_graph.set_title('Position_3D Plot')

    # Compare estimated position with ground truth
    plot_fig, plot_graph = plt.subplots(3, 1)
    plot_graph[0].plot(ground_truth_t, ground_truth[0])
    plot_graph[0].plot(est_t, estimated[0])
    plot_graph[0].set_ylabel('X_axis')

    plot_graph[1].plot(ground_truth_t, ground_truth[1])
    plot_graph[1].plot(est_t, estimated[1])
    plot_graph[1].set_ylabel('Y_axis')

    plot_graph[2].plot(ground_truth_t, ground_truth[2], label='Ground Truth')
    plot_graph[2].plot(est_t, estimated[2], label='Estimated')
    plot_graph[2].set_ylabel('Z_axis')
    plot_graph[2].set_xlabel('Time')
    plot_graph[2].legend()
    plot_graph[0].set_title('Position_Time Plot')

    # Compare estimated angle with ground truth
    plot_fig, plot_graph = plt.subplots(3, 1)
    plot_graph[2].plot(ground_truth_t, ground_truth[5], label='Ground Truth')
    plot_graph[2].plot(est_t, estimated[3], label='Estimated')
    plot_graph[2].set_ylabel('Yaw angle')
    plot_graph[2].legend()

    plot_graph[1].plot(ground_truth_t, ground_truth[4])
    plot_graph[1].plot(est_t, estimated[4])
    plot_graph[1].set_ylabel('Pitch angle')

    plot_graph[0].plot(ground_truth_t, ground_truth[3])
    plot_graph[0].plot(est_t, estimated[5])
    plot_graph[0].set_ylabel('Roll angle')
    plot_graph[0].set_title('Orientation_Time Plot')

    plt.show()

if __name__ == "__main__":
    filename = "data\data\studentdata0.mat"
    simulation(filename)