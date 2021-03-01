import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm # [pip install tqdm]
import torch # [conda install pytorch -c pytorch, only python 3!]
import photontorch as pt # [pip install photontorch] my simulation/optimization library

tau_stacked = 0.5
eta_stacked = 0.5
theta_t = np.pi / 3
theta_b = np.pi / 3

theta_b = theta_b
tau = np.sqrt(tau_stacked)
eta = np.sqrt(eta_stacked)
gamma = 1j * np.sqrt(1 - eta_stacked)
kappa = 1j * np.sqrt(1 - tau_stacked)





E0 = tau
E1 = kappa * np.exp(-1j * theta_b / 2)
E2 = -np.conjugate(kappa) * np.exp(-1j * theta_b / 2)
E3 = tau * np.exp(-1j * theta_b)

S_B = np.array([[0, E0, E1, 0], [E0, 0, 0, E1], [E2, 0, 0, E3], [0, E2, E3, 0]])

# Create 4x4 matrix SI/S2
E0 = eta
E1 = gamma
E2 = gamma
E3 = eta

S_I = np.array([[0, E0, E1, 0], [E0, 0, 0, E1], [E2, 0, 0, E3], [0, E2, E3, 0]])

S = S_B.dot(S_I)

print(S)