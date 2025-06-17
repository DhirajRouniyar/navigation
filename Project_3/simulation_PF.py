import observation_model as obs
import numpy as np
import plots
import time
from scipy.linalg import block_diag
import compute_covariance
import particle_filter



def size_of_map(estimated):
    # Returns : size of the map.
    # Get max and min values of the estimated
    maximum_x = np.max(estimated[0])
    minimum_x = np.min(estimated[0])
    maximum_y = np.max(estimated[1])
    minimum_y = np.min(estimated[1])
    maximum_z = np.max(estimated[2])
    minimum_z = np.min(estimated[2])

    map = np.array([maximum_x, minimum_x, maximum_y, minimum_y, maximum_z, minimum_z])

    buffer_time = 0.5        # buffer of 0.5 meters
    map[0] += buffer_time
    map[1] -= buffer_time
    map[2] += buffer_time
    map[3] -= buffer_time
    map[4] += buffer_time
    map[5] -= buffer_time

    return map

def simulate(filename: str, partcl_cnt = None ,method = "weighted_avg",R = None, Qa = None, Qg = None):
    
    model = obs.Observation_Model()
    data, ground_truth, ground_truth_t = model.read_data(filename)

    pos = []
    ort = []
    est_t = []
    data = data
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

    if R is None:
        covariance = compute_covariance.covariance_compute(estimated, est_t, ground_truth, ground_truth_t)
        R = covariance   

    num_partcl = partcl_cnt
    P_pos = 0.05 ** 2 * np.eye(3)
    P_rot = 0.05 ** 2 * np.eye(3)
    P_vel = 0.05 ** 2 * np.eye(3)
    P_bias = 0.05 * np.eye(6)
    P = block_diag(P_pos, P_rot, P_vel, P_bias) 

    map = size_of_map(estimated)
    prtcl_filt = particle_filter.ProbabilisticTracker(num_partcl, R, map, Qa, Qg)

    filt_data = np.zeros((6, len(estimated[0])))
    t = 0   
    x = ground_truth[:, :][:, 0]  
    i = 0   
    partcl = prtcl_filt.initialize_random_particles()
    x = np.concatenate((x, np.zeros(3)))
    partcl_hist = np.zeros((num_partcl, 6, len(estimated[0])))
    start = time.time()                     # Time taken for simulation
    for index, data_ in enumerate(data):
        p = None
        r = None
        if isinstance(data_['id'], int):
            p, r = model.estimate_pose([data_['id']], index)

        elif data_['id'].shape[0] > 1:
            p, r = model.estimate_pose(data_['id'], index)

        if p is not None:
            dt = data_["t"] - t

            u = np.concatenate((data_['omg'], data_['acc']), axis=0)    
            z = np.array([p[0], p[1], p[2], r[0], r[1], r[2], 0, 0, 0, 0, 0, 0, 0, 0, 0])   

            partcl = prtcl_filt.predict_step(partcl, u, dt)  
            wgts = prtcl_filt.update_step(partcl, z)          
            est = prtcl_filt.extract_pose(partcl, wgts, method)    
            partcl_hist[:, :, i] = partcl[:, :6, :].reshape((num_partcl, 6,))  
            partcl = prtcl_filt.resample_particles(partcl, wgts)   
            t = data_["t"]
            filt_data[:, i] = est[0:6].reshape((6,))   
            i = i + 1

    end = time.time()
    exe_t = end - start  

    return estimated, est_t, ground_truth, ground_truth_t, filt_data, partcl_hist, exe_t

if __name__ == "__main__":
    estimated, est_t, ground_truth, ground_truth_t, filt_data, partcl_hist, exe_t = simulate(r"data\data\studentdata1.mat", 1000, Qa=100, Qg=0.1)
    plots.plot(estimated, est_t, ground_truth, ground_truth_t, filt_data)

