import observation_model as obs_model
import numpy as np
import scipy.io as sio
from scipy.linalg import block_diag
import compute_covariance as com_cov
import ukf as ukf
import plots

def simulate(filename):

    model = obs_model.Observation_Model()
    data, ground_truth, ground_truth_t = model.read_data(filename)

    pos = []
    ort = []
    est_t = []
    
    for index, data_ in enumerate(data):
        if isinstance(data_['id'], int):
            p , r = model.estimate_pose([data_['id']], index)
            pos.append(p)
            ort.append(r)
            est_t.append(float(data_['t']))
            continue
        if data_['id'].shape[0] > 1:
            p , r = model.estimate_pose(data_['id'], index)
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
    covariance = com_cov.covariance_compute(estimated, est_t, ground_truth, ground_truth_t)

    R = covariance    # observation noise covariance matrix

    Q0 = 0.001 * np.eye(6, 6)   # process noise covariance matrix, where noise is the bias drift

    B = np.vstack((np.zeros((9, 6)), np.eye(6, 6)))

    # initial state covariance matrix
    P_pos = 0.05 ** 2 * np.eye(3)
    P_rot = 0.05 ** 2 * np.eye(3)
    P_vel = 0.05 ** 2 * np.eye(3)
    P_bias = 0.05 * np.eye(6)
    P = block_diag(P_pos, P_rot, P_vel, P_bias)   # initial state covariance matrix

    uk = ukf.UKF(P, Q0, R, 0.01)  # Unscented Kalman Filter object

    filtered_data = np.zeros((6, len(estimated[0])))

    t = 0   # initial time
    x = np.concatenate((ground_truth[0:6, :][:, 0], np.zeros(9)))   # initial state
    i = 0   # index for filtered data

    for index, data_ in enumerate(data):
        pos = None
        rot = None
        if isinstance(data_['id'], int):
            pos, rot = model.estimate_pose([data_['id']], index)

        elif data_['id'].shape[0] > 1:
            pos, rot = model.estimate_pose(data_['id'], index)

        if pos is not None:
            dt = data_["t"] - t
            try:
                u = np.concatenate((data_['omg'], data_['acc']), axis=0)
            except:
                u = np.concatenate((data_['drpy'], data_['acc']), axis=0)

            Q = (dt * B) @ Q0 @ (dt * B).T

            z = np.concatenate((pos, rot), axis=0)

            x, P = uk.predict(x, u, P, Q, dt)    # predict the next state
            x, P = uk.update(x, z, P, R)         # refine the state estimate

            t = data_["t"]

            filtered_data[:, i] = x[0:6]
            i = i + 1

    plots.plot(estimated, est_t, ground_truth, ground_truth_t, filtered_data)


# if __name__ == "__main__":
#     filename = "data\data\studentdata0.mat"
#     simulate(filename)