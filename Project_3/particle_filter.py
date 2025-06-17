import numpy as np

class ProbabilisticTracker:
    def __init__(self, particle_count, measurement_noise, bounds, acc_noise, gyro_noise):
        self.particle_count = particle_count
        self.bounds = bounds
        self.acc_noise = acc_noise
        self.gyro_noise = gyro_noise
        self.measurement_noise = measurement_noise
        self.min_particles = particle_count // 2
        self.state_samples = np.zeros((self.particle_count, 6))

    def initialize_random_particles(self):
        x_limits = (self.bounds[0], self.bounds[1])
        y_limits = (self.bounds[2], self.bounds[3])
        z_limits = (self.bounds[4], self.bounds[5])

        yaw_limits = (-np.pi / 2, np.pi / 2)
        pitch_limits = (-np.pi / 2, np.pi / 2)
        roll_limits = (-np.pi / 2, np.pi / 2)

        min_vals = np.array([x_limits[0], y_limits[0], z_limits[0], yaw_limits[0], pitch_limits[0], roll_limits[0]])
        max_vals = np.array([x_limits[1], y_limits[1], z_limits[1], yaw_limits[1], pitch_limits[1], roll_limits[1]])

        random_particles = np.random.uniform(low=min_vals, high=max_vals, size=(self.particle_count, 6))
        random_particles = np.concatenate((random_particles, np.zeros((self.particle_count, 9))), axis=1)
        random_particles = np.expand_dims(random_particles, axis=-1)

        return random_particles

    def initialize_gaussian_particles(self, mean_state, cov_matrix):
        gaussian_particles = np.random.multivariate_normal(mean_state, cov_matrix, self.particle_count)
        gaussian_particles = np.expand_dims(gaussian_particles, axis=-1)
        return gaussian_particles

    def compute_dynamics(self, state_batch, accel_input, gyro_input):
        dynamics = np.zeros((self.particle_count, 15, 1))
        roll, pitch, yaw = state_batch[:, 3], state_batch[:, 4], state_batch[:, 5]

        rot_matrix = np.zeros((self.particle_count, 3, 3, 1))
        rot_matrix[:, 0, 0] = np.cos(yaw) * np.cos(pitch) - np.sin(yaw) * np.sin(pitch) * np.sin(roll)
        rot_matrix[:, 0, 1] = -np.cos(roll) * np.sin(yaw)
        rot_matrix[:, 0, 2] = np.cos(yaw) * np.sin(pitch) + np.cos(pitch) * np.sin(roll) * np.sin(yaw)
        rot_matrix[:, 1, 0] = np.cos(pitch) * np.sin(yaw) + np.cos(yaw) * np.sin(roll) * np.sin(pitch)
        rot_matrix[:, 1, 1] = np.cos(yaw) * np.cos(roll)
        rot_matrix[:, 1, 2] = np.sin(yaw) * np.sin(pitch) - np.cos(yaw) * np.cos(pitch) * np.sin(roll)
        rot_matrix[:, 2, 0] = -np.cos(roll) * np.sin(pitch)
        rot_matrix[:, 2, 1] = np.sin(roll)
        rot_matrix[:, 2, 2] = np.cos(roll) * np.cos(pitch)
        rot_matrix = rot_matrix.reshape((self.particle_count, 3, 3))

        orient_matrix = np.zeros((self.particle_count, 3, 3, 1))
        orient_matrix[:, 0, 0] = np.cos(pitch)
        orient_matrix[:, 0, 2] = -np.cos(roll) * np.sin(pitch)
        orient_matrix[:, 1, 1] = 1
        orient_matrix[:, 1, 2] = np.sin(roll)
        orient_matrix[:, 2, 0] = np.sin(pitch)
        orient_matrix[:, 2, 2] = np.cos(roll) * np.cos(pitch)
        orient_matrix = orient_matrix.reshape((self.particle_count, 3, 3))

        dynamics[:, 0:3] = state_batch[:, 6:9]
        dynamics[:, 3:6] = np.linalg.inv(orient_matrix) @ (gyro_input - state_batch[:, 9:12])
        gravity = np.array([0, 0, -9.81]).reshape((3, 1))
        dynamics[:, 6:9] = gravity + rot_matrix @ (accel_input - state_batch[:, 12:15])

        return dynamics

    def predict_step(self, particles, control_input, time_step):
        perturbation = np.zeros((self.particle_count, 6, 1))
        perturbation[:, 0:3] = np.random.normal(scale=self.gyro_noise, size=(self.particle_count, 3, 1))
        perturbation[:, 3:6] = np.random.normal(scale=self.acc_noise, size=(self.particle_count, 3, 1))

        gyro_input = np.tile(control_input[:3].reshape(3, 1), (self.particle_count, 1, 1))
        acc_input = np.tile(control_input[3:6].reshape((3, 1)), (self.particle_count, 1, 1))

        acc_input += perturbation[:, 3:6]
        state_derivative = self.compute_dynamics(particles, acc_input, gyro_input)
        state_derivative[:, 3:6] += perturbation[:, 0:3]

        particles += state_derivative * time_step
        return particles

    def update_step(self, particles, observation):
        obs_matrix = np.zeros((6, 15))
        obs_matrix[0:6, 0:6] = np.identity(6)

        obs_noise = np.diag(self.measurement_noise).reshape((1, 6))
        observed_states = ((obs_matrix @ particles).reshape((self.particle_count, 6)) + obs_noise)
        observed_states = np.concatenate((observed_states, np.zeros((self.particle_count, 9))), axis=1)

        weights = self.calculate_weights(observed_states, observation)
        return weights

    def calculate_weights(self, predicted, actual):
        delta = predicted[:, 0:6] - actual[0:6]
        likelihood = np.exp(-0.5 * np.sum(delta ** 2, axis=1))
        normalized_weights = likelihood / np.sum(likelihood)
        return normalized_weights

    def resample_particles(self, particles, weights):
        new_particles = np.zeros((self.particle_count, 15, 1))
        weights /= np.sum(weights)
        cumulative = weights[0]
        index = 0
        start = np.random.uniform(0, 1 / self.particle_count)

        for i in range(self.particle_count):
            threshold = start + i / self.particle_count
            while cumulative < threshold:
                index += 1
                cumulative += weights[index]
            new_particles[i] = particles[index]
        return new_particles

    def extract_pose(self, particles, weights, strategy='weighted_avg'):
        if strategy == 'weighted_avg':
            pose = np.sum(particles * weights.reshape(self.particle_count, 1, 1), axis=0)
        elif strategy == 'highest_weight':
            pose = particles[np.argmax(weights)]
        elif strategy == 'average':
            pose = np.mean(particles, axis=0)
        return pose


