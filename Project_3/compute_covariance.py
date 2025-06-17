import numpy as np
import observation_model as obs

def covariance_compute(estimated, est_t, ground_truth, ground_truth_t):
   
    covariance = []
    for index, data_ in enumerate(estimated.T):
        # find the corresponding ground truth data
        data_idx_gt = np.argmin(np.abs(ground_truth_t - est_t[index]))
        gt_datum = ground_truth[:, data_idx_gt]
        gt_datum = np.array([gt_datum[0], gt_datum[1], gt_datum[2], gt_datum[3], gt_datum[4], gt_datum[5]])

        error = gt_datum.reshape(6, 1) - data_.reshape(6, 1)  # error
        cov = error @ error.T
        covariance.append(cov)

    average_cov = (1/(len(covariance)-1)) * np.sum(covariance, axis=0)   # average covariance matrix

    return average_cov

def estimate_covariances(filename):
   
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
    avg_cov = covariance_compute(estimated, est_t, ground_truth, ground_truth_t)

    return avg_cov

def average_covariance():
    
    filename = 'data/data/studentdata%d.mat'

    avg_cov= []
    for i in range(1, 8):
        file = filename % i
        avg_cov.append(estimate_covariances(file))

    avg_cov_allFiles = np.mean(avg_cov, axis=0)

    return avg_cov_allFiles



# if __name__ == "__main__":
#    Result = average_covariance()
#    print(Result)