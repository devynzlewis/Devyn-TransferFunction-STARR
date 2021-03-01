import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm # [pip install tqdm]
import torch # [conda install pytorch -c pytorch, only python 3!]
import photontorch as pt # [pip install photontorch] my simulation/optimization library


class DirectionalCoupler(pt.Component):
    r""" A directional coupler is a component with 4 ports that introduces no delays

    Each directional coupler has four ports. They are numbered 0 to 3:

    Ports:
       3        2
        \______/
        /------\
       0        1

    """

    # photontorch requires you to explicitly define the number of
    # ports in the component as a class variable:
    num_ports = 4

    def __init__(self, coupling=0.6, name=None):
        """ creation of a new waveguide """
        super(DirectionalCoupler, self).__init__(name=name)# always initialize parent first

        # to save the coupling as an optimizable parameter, we could just do the
        # same as we did for the waveguide: create a torch tensor and store it as a parameter:
        # coupling = torch.tensor(float(coupling))
        # self.phase = torch.nn.Parameter(data=coupling)

        # however, this could lead to problems, as this parameter would be unbounded
        # and we know for a fact the coupling should be bounded between 0 and 1.
        # an easy solution is to define the coupling as the cosine of a hidden parameter
        # which we call (with little imagination) `parameter`:

        # create a parameter. The coupling will be derived from the parameter as cos(self.parameter):
        parameter = torch.tensor(np.arccos(float(coupling)), dtype=torch.get_default_dtype())
        self.parameter = torch.nn.Parameter(data=parameter)

    @property
    def coupling(self):
        return torch.cos(self.parameter)

    def set_S(self, S):
        """ Fill the S-matrix with elements. Rememeber that the S-matrix has a shape

        (2=(real|imag), #wavelengths, #ports, #ports)

        """

        t = (1 - self.coupling) ** 0.5
        k = self.coupling ** 0.5

        # real part scattering matrix (transmission):
        S[0, :, 0, 1] = S[0, :, 1, 0] = t # same for all wavelengths
        S[0, :, 2, 3] = S[0, :, 3, 2] = t # same for all wavelengths

        # imag part scattering matrix (coupling):
        S[1, :, 0, 2] = S[1, :, 2, 0] = k # same for all wavelengths
        S[1, :, 1, 3] = S[1, :, 3, 1] = k # same for all wavelengths


class RBS(pt.Component):
    num_ports = 4

    def __init__(
        self,
        tau_stacked= 0.5,
        eta_stacked=1 / 2,
        theta_t= 0.2,
        theta_b= np.pi/3,
        trainable=False,
        name=None,
    ):
        super(RBS, self).__init__(name=name)

        self.theta_b = theta_b
        self.tau = np.sqrt(tau_stacked)
        self.eta = np.sqrt(eta_stacked)
        self.gamma = 1j * np.sqrt(1 - eta_stacked)
        self.kappa = 1j * np.sqrt(1 - tau_stacked)

        if not trainable:
            self.theta_t = theta_t
        else:
            theta_t = torch.tensor(theta_t, dtype=torch.float64)
            self.theta_t = torch.nn.Parameter(data=theta_t)

    def set_S(self, S):
        # Create 2x2 matrix SB/S1___________________________________________________________________________

        E0 = self.tau
        E1 = self.kappa * np.exp(-1j * self.theta_b / 2)
        E2 = -np.conjugate(self.kappa)*np.exp(-1j * self.theta_b/2)
        E3 = self.tau * np.exp(-1j * self.theta_b)

        S_B = np.array([[E0, E1], [E2, E3]])

        # Create 2x2 matrix SI/S2
        E0 = self.eta
        E1 = self.gamma
        E2 = self.gamma
        E3 = self.eta

        S_I = np.array([[E0, E1], [E2, E3]])

        # Create 2x2 matrix ST/S3
        E0 = self.tau * np.exp(-1j * self.theta_t)
        E1 = -np.conjugate(self.kappa)*np.exp(-1j * self.theta_t/2)
        E2 = self.kappa * np.exp(-1j * self.theta_t / 2)
        E3 = self.tau
        S_T = np.array([[E0, E1], [E2, E3]])


        A = S_B.dot(S_I)
        S_DB = A.dot(S_T)

        print(S_T)


        S[0, 0, 1, 0] = np.real(S_DB[0, 0])
        S[0, 0, 2, 0] = np.real(S_DB[1, 0])
        S[0, 0, 1, 3] = np.real(S_DB[0, 1])
        S[0, 0, 2, 3] = np.real(S_DB[1, 1])

        S[0, 0, 0, 1] = np.real(S_DB[0, 0])
        S[0, 0, 0, 2] = np.real(S_DB[0, 1])
        S[0, 0, 3, 1] = np.real(S_DB[1, 0])
        S[0, 0, 3, 2] = np.real(S_DB[1, 1])

        S[1, 0, 1, 0] = np.imag(S_DB[0, 0])
        S[1, 0, 2, 0] = np.imag(S_DB[1, 0])
        S[1, 0, 1, 3] = np.imag(S_DB[0, 1])
        S[1, 0, 2, 3] = np.imag(S_DB[1, 1])

        S[1, 0, 0, 1] = np.imag(S_DB[0, 0])
        S[1, 0, 0, 2] = np.imag(S_DB[0, 1])
        S[1, 0, 3, 1] = np.imag(S_DB[1, 0])
        S[1, 0, 3, 2] = np.imag(S_DB[1, 1])


class testCoupler(pt.Network):
    def __init__(self, name=None):
        super(testCoupler, self).__init__(name=name)

        self.source = pt.Source()
        self.detector1 = self.detector2 = self.detector3 = pt.Detector()
        self.dc1 = RBS()

        self.link('source:0', '0:dc1:1', '0:detector1')
        self.link('detector2:0', '3:dc1:2', '0:detector3')



dt = 1e-14 #[s]
total_time = 2000*dt #[s]
time = np.arange(0, total_time, dt)



env = pt.Environment(
    wavelength = 1e-6, #[m]
    freqdomain=False,
    time=time,# we will be doing frequency domain simulations
)

test = testCoupler()

# set the global simulation environment:
pt.set_environment(env)

detected = test(source=1)

test.plot(detected)
plt.show()