import numpy as np
from math import cos, sin
from scipy.linalg import sqrtm
from typing import Tuple
import matplotlib.pyplot as plt

class UKF:
    def __init__(self, P, Q, R, dt):
        self.P = P                # Initial state covariance
        self.Q = Q                # Process noise covariance
        self.R = R                # Observation noise covariance
        self.dt = dt              # Time Step
        self.state_dim = 15       # State dimension
        self.m = 15               # Number of sigma points per directions
        self.kappa = 1            # Tuning parameter
        self.alpha = 0.01         # Determines spread of sigma points
        self.beta = 2             # Incorporate prior knowledge of distribution (for Gaussian)


    def compute_sigma(self, x, P): 
        #Generate sigma points

        sz_ = x.shape[0]
        state_dim = self.state_dim
        sig_pnt = np.zeros((2 * state_dim + 1, sz_))
        sig_pnt[0] = x                               # A design parameter that lets one decide where in relation 

        l = self.alpha ** 2 * (sz_ + self.kappa) - sz_   # scaling factor
        covariance = sqrtm((state_dim + l) * P)

        for i in range(1, state_dim + 1):
            sig_pnt[i] = x + covariance[i - 1]
            sig_pnt[state_dim + i] = x - covariance[i - 1]

        return sig_pnt.T

    def compute_weights(self, x):
        # Compute weights.

        sz_ = x.size                                    # In some weights we consider the spread and prior knowledge of distribution
        l = self.alpha ** 2 * (sz_ + self.kappa) - sz_  # scaling factor
        Wgts_mean = np.zeros(2 * sz_ + 1) + 1 / (2 * (l + sz_))
        Wgts_cov = np.zeros(2 * sz_ + 1) + 1 / (2 * (l + sz_))

        Wgts_mean[0] = l / (sz_ + l)
        Wgts_cov[0] = l / (sz_ + l) + (1 - self.alpha ** 2 + self.beta)

        return Wgts_mean, Wgts_cov


    def compute_state(self,x,u):
        # Compute the state.

        phi, theta, psi = x[3:6]

        Rotation = np.zeros((3, 3))
        Rotation[0, 0] = cos(psi) * cos(theta) - sin(psi) * sin(theta) * sin(phi)
        Rotation[0, 1] = -cos(phi) * sin(psi)
        Rotation[0, 2] = cos(psi) * sin(theta) + cos(theta) * sin(phi) * sin(psi)
        Rotation[1, 0] = cos(theta) * sin(psi) + cos(psi) * sin(phi) * sin(theta)
        Rotation[1, 1] = cos(psi) * cos(phi)
        Rotation[1, 2] = sin(psi) * sin(theta) - cos(psi) * cos(theta) * sin(phi)
        Rotation[2, 0] = -cos(phi) * sin(theta)
        Rotation[2, 1] = sin(phi)
        Rotation[2, 2] = cos(phi) * cos(theta)

        G = np.zeros((3, 3))
        G[0, 0] = cos(theta)
        G[0, 2] = -cos(phi) * sin(theta)
        G[1, 1] = 1
        G[1, 2] = sin(phi)
        G[2, 0] = sin(theta)
        G[2, 2] = cos(phi) * cos(theta)

        xdot = np.zeros(x.shape)
        xdot[0:3] = x[6:9]
        xdot[3:6] = np.linalg.inv(G) @ (u[0:3] - x[9:12])
        xdot[6:9] = np.array([0, 0, -9.81]) + Rotation @ (u[3:6] - x[12:15])

        return xdot


    def calculate_observation(self,x: np.ndarray) -> np.ndarray:
        # Calculate the state observation.
        # observation matrix

        C = np.zeros((6, 15))
        C[0:6, 0:6] = np.eye(6, 6)

        return C @ x


    def predict(self,x: np.ndarray,
                u: np.ndarray,
                P: np.ndarray,
                Q: np.ndarray,
                dt: float) -> Tuple[np.ndarray, np.ndarray]:
    
        # Predict the next state.
        dim = x.size

        # calculate the sigma points and weights
        sig = self.compute_sigma(x, P)
        wgts_mean, wgts_cov = self.compute_weights(x)

        # propagate sigma points
        for i in range(2 * dim + 1):
            sig[:, i] += self.compute_state(sig[:, i], u) * dt

        # compute mean (x)
        x = np.sum(wgts_mean * sig, axis=1)

        # compute covariance (P)
        # P = np.copy(Q)
        d = sig - x[:,np.newaxis]
        P = d @ np.diag(wgts_cov) @ d.T + Q

        return x, P


    def update(self,x: np.ndarray,
               z: np.ndarray,
               P: np.ndarray,
               R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # Update the state.
        
        dim = x.size

        # calculate the sigma points
        sig = self.compute_sigma(x, P)
        wgts_mean, wgts_cov = self.compute_weights(x)

        # compute sigma for observation
        z_sigma = self.calculate_observation(sig)

        # compute observation mean
        z_mean = np.sum(wgts_mean * z_sigma, axis=1)

        # compute observation covariance
        # S = np.copy(R)
        dz = z_sigma - z_mean[:, np.newaxis]
        S = dz @ np.diag(wgts_cov) @ dz.T + R

        # compute cross covariance
        V = np.zeros((dim, z.size))
        dx = sig - x[:, np.newaxis]
        V += dx @ np.diag(wgts_cov) @ dz.T

        # update state mean and covariance
        K = V @ np.linalg.inv(S)   # Kalman gain
        x += K @ (z - z_mean)
        P -= K @ S @ K.T

        return x, P
        
## Line 44: In some weights we consider the spread and prior knowledge of distribution
## Line 26, 27 : # A design parameter that lets one decide where in relation  
                 # to the error ellipse the sigma points should be placed.