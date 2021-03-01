from random import seed
from random import random
seed(1)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K


def RISQ_Calc(theta_b,theta_t,theta_ap,eta_sqrd,tau_sqrd,tau_ap):
    N = len(theta_ap)
    U_f = np.zeros((2, 2, 5000), dtype=complex)

    for i in range(0, N):
        S_BS = generate_S_DB(theta_b, theta_t, eta_sqrd, tau_sqrd)

        U_f[:, :, i] = S_BS
    return U_f

def generate_S_DB(theta_b, theta_t, eta_sqrd, tau_sqrd):
    eta = np.sqrt(eta_sqrd)
    tau = np.sqrt(tau_sqrd)
    kappa = 1j * np.sqrt(1 - tau_sqrd)
    gamma = 1j * np.sqrt(1 - eta_sqrd)

    S_b = generate_S_b(theta_b, tau, kappa)

    return S_b


def generate_S_b(theta, tau, kappa):
    t_b = tau

    s_b = -np.conjugate(kappa)*np.exp(-1j * theta/2)

    sPrime_b = kappa*np.exp(-1j * theta/2)

    tPrime_b = tau*np.exp(-1j * theta)

    return np.array([[t_b, sPrime_b], [s_b, tPrime_b]])


tau_stacked = 0.4
eta_stacked = 1/2
theta_t = np.pi/3
theta_b = np.pi/3
tau_ap = 1/2
theta_ap = np.linspace(-np.pi, np.pi, 5000)

U_RISQ = RISQ_Calc(theta_b, theta_t, theta_ap, eta_stacked, tau_stacked, tau_ap)

P_a_c = abs((U_RISQ[0][0][:])**2)
P_a_l = abs((U_RISQ[0][1][:])**2)
dashedLine = np.zeros(5000)
dashedLine[:] = 0.5

plt.figure(1)
plt.plot(theta_ap, P_a_c, "-b", label="|\u03B1|^2")

plt.plot(theta_ap, P_a_l, "-g", label="|\u03B2|^2")
plt.plot(theta_ap, dashedLine, 'r--')

plt.show()

