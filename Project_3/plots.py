import matplotlib.pyplot as plt
import numpy as np

def plot(estimated, est_t, ground_truth, ground_truth_t, filtered_data):

    plot_fig = plt.figure()
    plot_graph = plot_fig.add_subplot(111, projection='3d')

    # Plot Graph steps
    plot_graph.plot(ground_truth[0], ground_truth[1], ground_truth[2], c='g', label='Ground Truth')
    plot_graph.plot(estimated[0], estimated[1], estimated[2], 'r--', label='Estimated')
    plot_graph.plot(filtered_data[0], filtered_data[1], filtered_data[2], label='Filtered')
    plot_graph.legend()
    plot_graph.set_xlabel('X_axis')
    plot_graph.set_ylabel('Y_axis')
    plot_graph.set_zlabel('Z_Axis')
    plot_graph.set_title('Position_3D Plot')

    # Estimated position with ground truth
    plot_fig, plot_graph = plt.subplots(3, 1)
    plot_graph[0].plot(ground_truth_t, ground_truth[0])
    plot_graph[0].plot(est_t, estimated[0])
    plot_graph[0].plot(est_t, filtered_data[0])
    plot_graph[0].set_ylabel('X_axis')

    plot_graph[1].plot(ground_truth_t, ground_truth[1])
    plot_graph[1].plot(est_t, estimated[1])
    plot_graph[1].plot(est_t, filtered_data[1])
    plot_graph[1].set_ylabel('Y_axis')

    plot_graph[2].plot(ground_truth_t, ground_truth[2], label='Ground Truth')
    plot_graph[2].plot(est_t, estimated[2], label='Estimated')
    plot_graph[2].plot(est_t, filtered_data[2], label='Filtered')
    plot_graph[2].set_ylabel('Z_axis')
    plot_graph[2].set_xlabel('Time')
    plot_graph[2].legend()
    plot_graph[0].set_title('Position_Time Plot')

    # Estimated angle with ground truth
    plot_fig, plot_graph = plt.subplots(3, 1)
    plot_graph[2].plot(ground_truth_t, ground_truth[5], label='Ground Truth')
    plot_graph[2].plot(est_t, estimated[3], label='Estimated')
    plot_graph[2].plot(est_t, filtered_data[5], label='Filtered')
    plot_graph[2].set_ylabel('Yaw angle')
    plot_graph[2].legend()

    plot_graph[1].plot(ground_truth_t, ground_truth[4])
    plot_graph[1].plot(est_t, estimated[4])
    plot_graph[1].plot(est_t, filtered_data[4])
    plot_graph[1].set_ylabel('Pitch angle')

    plot_graph[0].plot(ground_truth_t, ground_truth[3])
    plot_graph[0].plot(est_t, estimated[5])
    plot_graph[0].plot(est_t, filtered_data[3])
    plot_graph[0].set_ylabel('Roll angle')
    plot_graph[0].set_title('Orientation_Time Plot')
    
    a = rmse_loss(estimated, est_t, ground_truth, ground_truth_t, filtered_data)

    plt.show()

def rmse_loss(estimated, est_t, ground_truth, ground_truth_t, filtered_data):
    
    rmse_est = np.zeros(len(estimated[0]))
    rmse_filter = np.zeros(len(filtered_data[0]))

    for i in range(len(estimated[0])):
        data_idx_gt = np.argmin(np.abs(ground_truth_t - est_t[i]))
        rmse_est[i] = np.sqrt(np.mean((estimated[:3, i] - ground_truth[:3, data_idx_gt]) ** 2))
        rmse_filter[i] = np.sqrt(np.mean((filtered_data[:3, i] - ground_truth[:3, data_idx_gt]) ** 2))

    plot_fig = plt.figure()
    plot_graph = plot_fig.add_subplot(111)
    plot_graph.plot(est_t, rmse_est, label='Estimated RMSE')
    plot_graph.plot(est_t, rmse_filter, label='Filtered RMSE')
    plot_graph.legend()
    plot_graph.set_xlabel('Time')
    plot_graph.set_ylabel('RMSE')
    plot_graph.set_title('RMSE Loss')
    
    return plot_graph
