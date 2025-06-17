from __future__ import division
import numpy as np
import slam_utils
import tree_extraction
from scipy.stats.distributions import chi2
import sys
    

def vehicle_motion_model(control_input, time_step, ekf_state, vehicle_properties):
    '''
    Calculates the discretized motion model for the given vehicle, along with its Jacobian matrix

    Returns:
        motion_delta, a 3x1 vector representing the change in motion (x_{t+1} - x_t) given the control input.

        jacobian, the 3x3 Jacobian matrix of the motion model with respect to the vehicle state (x, y, phi)
    '''

    ###
    # Implement the vehicle motion model and its Jacobian as derived.
    ###
    velocity = control_input[0]
    steering_angle = slam_utils.wrap_angle(control_input[1])

    wheelbase = vehicle_properties['H']
    axle_length = vehicle_properties['L']
    front_axle_offset = vehicle_properties['a']
    rear_axle_offset = vehicle_properties['b']

    corrected_velocity = velocity / (1 - np.tan(steering_angle) * wheelbase / axle_length)
    
    # vehicle orientation = ekf_state['x'][2]
    orientation = slam_utils.wrap_angle(ekf_state['x'][2])

    motion_delta = np.empty([3], dtype=np.float64)
    
    motion_delta[0] = time_step * (corrected_velocity * np.cos(orientation) - corrected_velocity / axle_length * np.tan(steering_angle) * (front_axle_offset * np.sin(orientation) + rear_axle_offset * np.cos(orientation)))
    motion_delta[1] = time_step * (corrected_velocity * np.sin(orientation) + corrected_velocity / axle_length * np.tan(steering_angle) * (front_axle_offset * np.cos(orientation) - rear_axle_offset * np.sin(orientation)))
    motion_delta[2] = time_step * corrected_velocity / axle_length * np.tan(steering_angle)

    jacobian = np.empty([3, 3], dtype=np.float64)
    jacobian[0, 0] = 1
    jacobian[0, 1] = 0
    jacobian[0, 2] = time_step * (-corrected_velocity * np.sin(orientation) - corrected_velocity / axle_length * np.tan(steering_angle) * (front_axle_offset * np.cos(orientation) - rear_axle_offset * np.sin(orientation)))
    jacobian[1, 0] = 0
    jacobian[1, 1] = 1
    jacobian[1, 2] = time_step * (corrected_velocity * np.cos(orientation) + corrected_velocity / axle_length * np.tan(steering_angle) * (-front_axle_offset * np.sin(orientation) - rear_axle_offset * np.cos(orientation)))
    jacobian[2, 0] = 0
    jacobian[2, 1] = 0
    jacobian[2, 2] = 1

    return motion_delta, jacobian


def predict_odometry(control_input, time_step, ekf_state, vehicle_properties, process_noise_sigmas):
    '''
    Perform the prediction step of the EKF filter given an odometry measurement and time step, 
    where control_input = (velocity, steering_angle) as shown in the vehicle/motion model.

    Returns the updated ekf_state.
    '''

    ###
    # Implement the prediction propagation
    ###
    motion_delta, jacobian_matrix = vehicle_motion_model(control_input, time_step, ekf_state, vehicle_properties)

    state_covariance = ekf_state['P']    
    process_noise = np.diag([process_noise_sigmas['xy']**2, process_noise_sigmas['xy']**2, process_noise_sigmas['phi']**2])
    updated_state = ekf_state['x'][0:3] + motion_delta
    updated_covariance = np.matmul(np.matmul(jacobian_matrix, state_covariance[0:3, 0:3]), np.transpose(jacobian_matrix)) + process_noise

    ekf_state['x'][0:3] = updated_state
    ekf_state['x'][2] = slam_utils.wrap_angle(ekf_state['x'][2])
    
    ekf_state['P'][0:3, 0:3] = updated_covariance
    ekf_state['P'] = slam_utils.symmetrize(ekf_state['P'])
    
    return ekf_state




def update_with_gps(gps_measurement, ekf_state, process_noise_sigmas):
    '''
    Perform a measurement update of the EKF state given a GPS measurement (x, y).

    Returns the updated ekf_state.
    '''
    
    ###
    # Implement the GPS measurement update.
    ###
    residual = np.array(gps_measurement - ekf_state['x'][0:2])

    state_covariance = ekf_state['P']
    gps_noise = np.diag([process_noise_sigmas['gps']**2, process_noise_sigmas['gps']**2])
    measurement_matrix = np.zeros((2, ekf_state['x'].size))
    measurement_matrix[0, 0] = 1
    measurement_matrix[1, 1] = 1

    innovation_covariance_inv = np.linalg.inv(state_covariance[0:2, 0:2] + gps_noise)

    mahalanobis_distance = np.matmul(np.matmul(np.transpose(residual), innovation_covariance_inv), residual)
    
    if mahalanobis_distance < chi2.ppf(0.999, df=2):
        kalman_gain = np.matmul(np.matmul(state_covariance, np.transpose(measurement_matrix)), innovation_covariance_inv)
        ekf_state['x'] = ekf_state['x'] + np.matmul(kalman_gain, residual)
        ekf_state['x'][2] = slam_utils.wrap_angle(ekf_state['x'][2])
        temp_matrix = np.identity(ekf_state['x'].size) - np.matmul(kalman_gain, measurement_matrix)
        ekf_state['P'] = slam_utils.symmetrize(np.matmul(temp_matrix, state_covariance))

    return ekf_state



def compute_laser_measurement_model(ekf_state, landmark_index):
    ''' 
    Computes the expected measurement and Jacobian for a (range, bearing) sensor 
    observing the landmark with the given landmark_index.

    Returns:
        predicted_obs: the 2x1 expected measurement vector [range_hat, bearing_hat].

        measurement_jacobian: a 2 x (3 + 2 * num_landmarks) Jacobian matrix of the observation 
        with respect to the full EKF state vector.
    '''
    
    ###
    # Compute the predicted observation and Jacobian
    ###
    veh_x, veh_y, heading = ekf_state['x'][0:3]

    landmark_x, landmark_y = ekf_state['x'][1 + 2 * (landmark_index + 1) : 3 + 2 * (landmark_index + 1)]

    delta_x = landmark_x - veh_x
    delta_y = landmark_y - veh_y
    range_hat = np.sqrt(delta_x**2 + delta_y**2)
    bearing_hat = np.arctan2(delta_y, delta_x) - heading
    predicted_obs = (range_hat, slam_utils.wrap_angle(bearing_hat))

    measurement_jacobian = np.zeros((2, ekf_state['x'].size)) 

    range_sq = delta_x**2 + delta_y**2
    range_val = np.sqrt(range_sq)

    # Derivatives w.r.t. vehicle state (x, y, phi)
    measurement_jacobian[0, 0] = -delta_x / range_val
    measurement_jacobian[1, 0] =  delta_y / range_sq
    measurement_jacobian[0, 1] = -delta_y / range_val
    measurement_jacobian[1, 1] = -delta_x / range_sq
    measurement_jacobian[0, 2] = 0
    measurement_jacobian[1, 2] = -1

    # Derivatives w.r.t. landmark state
    landmark_x_index = 1 + 2 * (landmark_index + 1)
    landmark_y_index = 2 + 2 * (landmark_index + 1)

    measurement_jacobian[0, landmark_x_index] = delta_x / range_val
    measurement_jacobian[1, landmark_x_index] = -delta_y / range_sq
    measurement_jacobian[0, landmark_y_index] = delta_y / range_val
    measurement_jacobian[1, landmark_y_index] =  delta_x / range_sq

    return predicted_obs, measurement_jacobian


def add_new_landmark_to_state(ekf_state, tree_observation):
    '''
    Initializes a newly detected landmark in the EKF state, increasing the state 
    and covariance matrix dimensions accordingly.

    Returns the updated ekf_state.
    '''

    ###
    # Compute landmark position in global frame and expand the state and covariance
    ###
    veh_x, veh_y, heading = ekf_state['x'][0:3]
    current_covariance = ekf_state['P']
    range_measurement, bearing_measurement, tree_diameter = tree_observation
    bearing_measurement = slam_utils.wrap_angle(bearing_measurement)

    landmark_x = veh_x + range_measurement * np.cos(bearing_measurement + heading)
    landmark_y = veh_y + range_measurement * np.sin(bearing_measurement + heading)

    expanded_state = np.append(ekf_state['x'], [landmark_x, landmark_y])
    new_state_vector = np.array(expanded_state)

    new_covariance = np.zeros((new_state_vector.size, new_state_vector.size))
    new_covariance[:current_covariance.shape[0], :current_covariance.shape[1]] = current_covariance
    new_covariance[-2, -2] = 100  # Large initial uncertainty in landmark x
    new_covariance[-1, -1] = 100  # Large initial uncertainty in landmark y

    ekf_state['num_landmarks'] += 1
    ekf_state['x'] = new_state_vector
    ekf_state['P'] = slam_utils.symmetrize(new_covariance)

    return ekf_state



def compute_data_association(ekf_state, measurements, sigmas, params):
    '''
    Computes measurement data association.

    Given a robot and map state and a set of (range,bearing) measurements,
    this function should compute a good data association, or a mapping from 
    measurements to landmarks.

    Returns an array 'assoc' such that:
        assoc[i] == j if measurement i is determined to be an observation of landmark j,
        assoc[i] == -1 if measurement i is determined to be a new, previously unseen landmark, or,
        assoc[i] == -2 if measurement i is too ambiguous to use and should be discarded.
    '''

    if ekf_state["num_landmarks"] == 0:
        # set association to init new landmarks for all measurements
        return [-1 for m in measurements]

    ###
    # Implement this function.
    ###
    # print(ekf_state["num_landmarks"])

    P = ekf_state['P']
    R = np.diag([sigmas['range']**2, sigmas['bearing']**2])

    A = np.full((len(measurements),len(measurements)),chi2.ppf(0.96, df=2))
    cost_mat = np.full((len(measurements), ekf_state['num_landmarks']), chi2.ppf(0.96, df=2))

    for k in range(0,len(measurements)):
        for j in range(0,ekf_state['num_landmarks']):
            z_hat,H = compute_laser_measurement_model(ekf_state, j)
            # print(measurements[k][0:2])
            r = np.array(np.array(measurements[k][0:2]) - np.array(z_hat))
            S_inv = np.linalg.inv(np.matmul(np.matmul(H,P),np.transpose(H)) + R)
            MD = np.matmul(np.matmul(np.transpose(r),S_inv), r)
            cost_mat[k,j] = MD

    cost_mat_conc = np.concatenate((cost_mat, A), axis=1)        
    temp1 = np.copy(cost_mat)
    results = slam_utils.greedy_match(temp1)

    assoc = np.zeros(len(measurements),dtype = np.int32)
    for k in range(0, len(results)):
        # print(cost_mat[results[k][0],results[k][1]])
        if cost_mat_conc[results[k][0],results[k][1]] > chi2.ppf(0.99, df=2):
            assoc[results[k][0]] = -1
        elif cost_mat_conc[results[k][0],results[k][1]] >= chi2.ppf(0.95, df=2):
            assoc[results[k][0]] = -2
        else:
            assoc[results[k][0]] = results[k][1]

    return assoc


def laser_update(trees, assoc, ekf_state, sigmas, params):
    '''
    Perform a measurement update of the EKF state given a set of tree measurements.

    trees is a list of measurements, where each measurement is a tuple (range, bearing, diameter).

    assoc is the data association for the given set of trees, i.e. trees[i] is an observation of the
    ith landmark. If assoc[i] == -1, initialize a new landmark with the function add_new_landmark_to_state
    in the state for measurement i. If assoc[i] == -2, discard the measurement as 
    it is too ambiguous to use.

    The diameter component of the measurement can be discarded.

    Returns the ekf_state.
    '''

    ###
    # Implement the EKF update for a set of range, bearing measurements.
    ###

    R = np.diag([sigmas['range']**2, sigmas['bearing']**2])

    for i in range(0,len(trees)):
        if assoc[i]== -2:
            continue
        elif assoc[i]== -1:
            ekf_state = add_new_landmark_to_state(ekf_state,trees[i])
            P = ekf_state['P']
            z_hat,H = compute_laser_measurement_model(ekf_state, ekf_state['num_landmarks']-1)
            r = np.array(np.array(trees[i][0:2]) - np.array(z_hat))
            S_inv = np.linalg.inv(np.matmul(np.matmul(H,P),np.transpose(H)) + R) 
            K = np.matmul(np.matmul(P,np.transpose(H)),S_inv)
            ekf_state['x'] = ekf_state['x'] + np.matmul(K,r)
            ekf_state['x'][2] = slam_utils.wrap_angle(ekf_state['x'][2])
            temp1 = np.identity(P.shape[0]) - np.matmul(K,H)
            ekf_state['P'] =  slam_utils.symmetrize(np.matmul(temp1,P))
        else:
            P = ekf_state['P']
            z_hat,H = compute_laser_measurement_model(ekf_state,assoc[i])
            r = np.array(np.array(trees[i][0:2]) - np.array(z_hat))
            S_inv = np.linalg.inv(np.matmul(np.matmul(H,P),np.transpose(H)) + R)
            K = np.matmul(np.matmul(P,np.transpose(H)),S_inv)
            ekf_state['x'] = ekf_state['x'] + np.matmul(K,r)
            ekf_state['x'][2] = slam_utils.wrap_angle(ekf_state['x'][2])
            temp2 = np.identity(P.shape[0]) - np.matmul(K,H)
            ekf_state['P'] =  slam_utils.symmetrize(np.matmul(temp2,P))

    return ekf_state


def run_ekf_slam(events, ekf_state_0, vehicle_params, filter_params, sigmas):
    last_odom_t = -1
    ekf_state = {
        'x': ekf_state_0['x'].copy(),
        'P': ekf_state_0['P'].copy(),
        'num_landmarks': ekf_state_0['num_landmarks']
    }
    
    state_history = {
        't': [0],
        'x': ekf_state['x'],
        'P': np.diag(ekf_state['P'])
    }

    if filter_params["do_plot"]:
        plot = slam_utils.setup_plot()

    for i, event in enumerate(events):
        t = event[1][0]
        if i % 1000 == 0:
            print("t = {}".format(t))

        if event[0] == 'gps':
            gps_msmt = event[1][1:]
            ekf_state = update_with_gps(gps_msmt, ekf_state, sigmas)

        elif event[0] == 'odo':
            if last_odom_t < 0:
                last_odom_t = t
                continue
            u = event[1][1:]
            dt = t - last_odom_t
            ekf_state = predict_odometry(u, dt, ekf_state, vehicle_params, sigmas)
            last_odom_t = t

        else:
            # Laser
            scan = event[1][1:]
            trees = tree_extraction.extract_trees(scan, filter_params)
            assoc = compute_data_association(ekf_state, trees, sigmas, filter_params)
            ekf_state = laser_update(trees, assoc, ekf_state, sigmas, filter_params)
            if filter_params["do_plot"]:
                slam_utils.update_display(state_history['x'], ekf_state, trees, scan, assoc, plot, filter_params,i)

        
        state_history['x'] = np.vstack((state_history['x'], ekf_state['x'][0:3]))
        state_history['P'] = np.vstack((state_history['P'], np.diag(ekf_state['P'][:3,:3])))
        state_history['t'].append(t)

    return state_history


def main():
    odo = slam_utils.load_data("data/DRS.txt")
    gps = slam_utils.load_data("data/GPS.txt")
    laser = slam_utils.load_data("data/LASER.txt")

    # collect all events and sort by time
    events = [('gps', x) for x in gps]
    events.extend([('laser', x) for x in laser])
    events.extend([('odo', x) for x in odo])

    events = sorted(events, key = lambda event: event[1][0])

    vehicle_params = {
        "a": 3.78,
        "b": 0.50, 
        "L": 2.83,
        "H": 0.76
    }

    filter_params = {
        # measurement params
        "max_laser_range": 75, # meters

        # general...
        "do_plot": True,
        "plot_raw_laser": True,
        "plot_map_covariances": True

        # Add other parameters here if you need to...
    }

    # Noise values
    sigmas = {
        # Motion model noise
        "xy": 0.05,
        "phi": 0.5*np.pi/180,

        # Measurement noise
        "gps": 3,
        "range": 0.5,
        "bearing": 5*np.pi/180
    }

    # Initial filter state
    ekf_state = {
        "x": np.array( [gps[0,1], gps[0,2], 36*np.pi/180]),
        "P": np.diag([.1, .1, 1]),
        "num_landmarks": 0
    }

    run_ekf_slam(events, ekf_state, vehicle_params, filter_params, sigmas)

if __name__ == '__main__':
    main()
