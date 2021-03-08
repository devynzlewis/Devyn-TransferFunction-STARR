import numpy as np
import matplotlib.pyplot as plt


# ________________________________________________________________________________________


def RISQ_Calc(theta_b,theta_t,theta_ap,eta_sqrd,tau_sqrd,tau_ap):
    N = len(theta_ap)
    U_f = np.zeros((2, 2, 5000), dtype=complex)
    zeroangle = np.zeros(5000)

    for i in range(0, N):
        S_BS1 = generate_S_DB(theta_b, theta_t, eta_sqrd, tau_sqrd)
        S_BS2 = generate_S_DB(theta_b, theta_t, eta_sqrd, tau_sqrd)
        S_phase_phi = generate_S_AP(theta_ap[i], tau_ap)
        S_phase_theta = generate_S_AP(zeroangle[i], tau_ap)

        A = S_BS1.dot(S_phase_phi)
        B = A.dot(S_BS2)
        U_f[:, :, i] = B.dot(S_phase_theta)



    return U_f
#-------------------------RISQ_Calc DEPENDS ON ALL--------------------------------------------------------------------------


def generate_S_DB(theta_b, theta_t, eta_sqrd, tau_sqrd):
    eta = np.sqrt(eta_sqrd)
    tau = np.sqrt(tau_sqrd)
    kappa = 1j * np.sqrt(1 - tau_sqrd)
    gamma = 1j * np.sqrt(1 - eta_sqrd)

    T_b = generate_transfer(generate_S_b(theta_b, tau, kappa))
    T_t = generate_transfer(generate_S_t(theta_t, tau, kappa))
    T_I = generate_transfer(generate_S_I(eta, gamma))

    A = T_b.dot(T_I)
    S_DB = A.dot(T_t)

    return np.transpose(generate_scattering(S_DB))


#-------------------S_DB DEPENDS ON THESE----------------------------------------------------------------------
def generate_transfer(S):
    a = S[0][0]
    b = S[0][1]
    c = S[1][0]
    d = S[1][1]

    det = np.linalg.det(S)

    return np.array([[1/c, -d/c], [a/c, -det/c]])


def generate_scattering(T):
    a = T[0][0]
    b = T[0][1]
    c = T[1][0]
    d = T[1][1]

    det = np.linalg.det(T)

    return np.array([[c / a, det / a], [1 / a, -b / a]])


def generate_S_b(theta, tau, kappa):
    t_b = tau

    s_b = -np.conjugate(kappa)*np.exp(-1j * theta/2)

    sPrime_b = kappa*np.exp(-1j * theta/2)

    tPrime_b = tau*np.exp(-1j * theta)

    return np.array([[t_b, sPrime_b], [s_b, tPrime_b]])


def generate_S_t(theta, tau, kappa):
    t_t = tau*np.exp(-1j * theta)

    sPrime_t = -np.conjugate(kappa) * np.exp(-1j * theta/2)

    s_t = kappa*np.exp(-1j * theta/2)

    tPrime_t = tau

    return np.array([[t_t, sPrime_t], [s_t, tPrime_t]])


def generate_S_I(eta, gamma):
    t_I = eta
    sPrime_I = gamma
    s_I = gamma
    tPrime_I = eta

    return np.array([[t_I, sPrime_I], [s_I, tPrime_I]])


#--------------------------------------------------------------------------------------------------------------------

def generate_S_AP(theta, tau):
    # kappa = 1j * np.sqrt(1-tau**2)

    kappa = 1j * np.sqrt(1 - tau**2)
    # tau = np.sqrt(tau)



    a_c = np.conjugate(1 / (tau - (abs(kappa)**2) * np.exp(-1j * theta) / (1 - tau * np.exp(-1j * theta))))


    return np.array([[1, 0], [0, a_c]])


def plotRISQCurve(theta_ap, P_a_c, P_a_l, dashedLine):
    plt.figure(1)
    plt.plot(theta_ap, P_a_c, "-b", label="|\u03B1|^2")

    plt.plot(theta_ap, P_a_l, "-g", label="|\u03B2|^2")

    plt.plot(theta_ap, dashedLine, 'r--')

    plt.legend(loc="upper right")
    plt.title("Output State Probabilities for a |0> Input State")
    plt.xlabel("All-Pass Ring Resonator Detuning \u03A6")
    plt.ylabel("Probability of Photon Measurement")

    plt.show()







#-------------------------------------RISQ_MODEL.m---------------------------------------------------------------------------------------------------------
# Initialize parameters

tau_stacked = 1/2
eta_stacked = 1/2
theta_t = np.pi/3
theta_b = np.pi/3
tau_ap = 1/2
theta_ap = np.linspace(-np.pi, np.pi, 5000)
N = 5000
dashedLine = np.ones(5000) * 0.5

U_RISQ = RISQ_Calc(theta_b, theta_t, theta_ap, eta_stacked, tau_stacked, tau_ap)



#%% ____________________________________________Output from mode a input (i.e., |0>)__________________________________________________________________________

P_a_c = abs((U_RISQ[0][0][:])**2)
P_a_l = abs((U_RISQ[0][1][:])**2)

# definining phase function from mode a to c for a single photon input

phase_P_a_c = np.reshape(np.angle(U_RISQ[0, 0, :]), [N,])
phase_P_a_c = - np.unwrap(np.degrees(phase_P_a_c))
#
# defining phase function from mode a to l for a single photon input
phase_P_a_l = np.reshape(np.angle(U_RISQ[0, 1, :]), [N,])
phase_P_a_l = np.degrees(phase_P_a_l)

plotRISQCurve(theta_ap, P_a_c, P_a_l, dashedLine)

# ______________________________________________________________________________________






