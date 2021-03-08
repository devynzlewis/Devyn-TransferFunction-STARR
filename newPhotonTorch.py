import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm # [pip install tqdm]
import torch # [conda install pytorch -c pytorch, only python 3!]
import photontorch as pt # [pip install photontorch] my simulation/optimization library

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

class Waveguide(pt.Component):
    """ Waveguide

    Each waveguides has two ports. They are numbered 0 and 1:

    Ports:

        0 ---- 1

    """

    # photontorch requires you to explicitly define the number of
    # ports in the component as a class variable:
    num_ports = 2

    def __init__(
        self,
        length=1e-5,
        loss=0, # in dB/m
        neff=2.34, # effective index of the waveguide
        ng=3.40, # group index of the waveguide
        wl0=1.55e-6, # center wavelength for which the waveguide is defined
        phase=0, # additional phase PARAMETER added to the waveguide
        trainable=True, # a flag to make the the component trainable or not
        name=None, # name of the waveguide
    ):
        """ creation of a new waveguide """
        super(Waveguide, self).__init__(name=name)# always initialize parent first
        # Handle inputs
        self.loss = float(loss)
        self.neff = float(neff)
        self.wl0 = float(wl0)
        self.ng = float(ng)
        self.length = float(length)


        # handle phase input
        phase = float(phase) % (2*np.pi)
        if not trainable: # if the network is not trainable, just store it as a normal float:
            self.phase = phase
        else: # else, make an optimizable parameter out of it:
            # create a torch tensor from the phase
            phase = torch.tensor(phase, dtype=torch.float64)
            # store the phase as a optimizable parameter
            self.phase = torch.nn.Parameter(data=phase)

    def set_delays(self, delays):
        """ set the delays for time-domain simulations """
        delays[:] = self.ng * self.length / self.env.c

    def set_S(self, S):
        """ set the S-matrix

        NOTE: because PyTorch does not support complex tensors, the real
        ane imaginary part of the S-matrix are stored in an extra dimension

        NOTE2: the S-matrix needs to be defined for all wavelengths, therefore
        one needs an extra dimension to store each different S-matrix for each
        wavelength

        ----------------

        Taking the above two notes into account, the S-matrix is thus a
        4-D tensor with shape

        (2=(real|imag), #wavelengths, #ports, #ports)

        """
        # during a photontorch simulation, the simulation environment
        # containing all the global simulation parameters will be
        # available to you as `self.env`:
        current_simulation_environment = self.env

        # you can use this environment to get information about the
        # wavelengths used in the simulation:
        wavelength = current_simulation_environment.wavelength

        # however, this wavelength is stored as a numpy array, while
        # photontorch expect torch tensors. We need to make a torch
        # tensor out of this:
        wavelength = torch.tensor(
            wavelength, # make this numpy array into a torch tensor
            dtype=torch.float64, # keep float64 dtype
            device=self.device, # put it on the current device ('cpu' or 'gpu')
        )

        # next we implement the dispersion, which will depend on the
        # wavelength tensor
        neff = self.neff - (wavelength - self.wl0) * (self.ng - self.neff) / self.wl0

        # we have now calculated an neff for each different wavelength.
        # let's calculate the phase depending on this neff:
        phase = (2 * np.pi * neff * self.length / wavelength) % (2 * np.pi)

        # next, we add the phase correction parameter.
        phase = phase + self.phase
        # note that in pytorch, inplace operations, such as
        # phase += self.phase
        # are not allowed, as they obscure the computation graph necessary to
        # perform the backpropagation algorithm later on...

        # because pytorch does not allow complex numbers, we split up exp(1j*phase) into
        # its real and imaginary part and revert back to the default dtype (usually float32).
        cos_phase = torch.cos(phase).to(torch.get_default_dtype())
        sin_phase = torch.sin(phase).to(torch.get_default_dtype())

        # finally, we can calculate the loss and add it to the phase, which
        # gives us the S-matrix parameters
        loss = 10 ** (-self.loss * self.length / 20)  # 20 because loss works on power
        re = loss * cos_phase
        ie = loss * sin_phase

        # the last thing to do is to add the S-matrix parameters to the S-matrix:
        S[0, :, 0, 1] = S[0, :, 1, 0] = re
        S[1, :, 0, 1] = S[1, :, 1, 0] = ie


class Crow(pt.Network):
    def __init__(
        self,
        num_rings=1,
        ring_length=1e-5, #[m]
        loss=1000, #[dB/m]
        neff=2.34,
        ng=3.4,
        wl0=1.55e-6,
        random_parameters=False,
        name=None
    ):
        super(Crow, self).__init__(name=name) # always initialize parent first

        # handle variables
        self.num_rings = int(num_rings)

        # define source and detectors:
        self.source = pt.Source()
        self.through = pt.Detector()
        self.drop = pt.Detector()
        self.add = pt.Detector()

        # if the random_parameters flag is set, we will initialize with
        # random parameters, else, we will initialize with parameters
        # set to zero:
        random_coupling = np.random.rand if random_parameters else (lambda : 0.5)
        random_phase = (lambda : 2*np.pi*np.random.rand()) if random_parameters else (lambda :0)

        # define directional couplers
        for i in range(self.num_rings + 1):
            self.add_component(
                name="dc%i"%i,
                comp=DirectionalCoupler(
                    coupling=random_coupling(), # initialize with random coupling
                )
            )

        # define waveguides between directional couplers:
        # let's only make the bottom waveguide trainable.
        for i in range(self.num_rings):
            self.add_component(
                name="top_wg%i"%i,
                comp=Waveguide(
                    length=0.5*ring_length,
                    loss=loss,
                    neff=neff,
                    ng=ng,
                    wl0=wl0,
                    phase=0,
                    trainable=False,
                )
            )
            self.add_component(
                name="btm_wg%i"%i,
                comp=Waveguide(
                    length=0.5*ring_length,
                    loss=loss,
                    neff=neff,
                    ng=ng,
                    wl0=wl0,
                    phase=random_phase(), # initialize with random phase
                    trainable=True,
                )
            )

        # lets now define the links
        link1 = ["source:0"]
        link2 = ["through:0"]
        for i in range(self.num_rings):
            link1 += ["0:dc%i:3"%i, "0:top_wg%i:1"%i]
            link2 += ["1:dc%i:2"%i, "0:btm_wg%i:1"%i]

        if self.num_rings % 2 == 1: # top=drop, btm=add
            link1 += ["0:dc%i:3"%(self.num_rings), "0:drop"]
            link2 += ["1:dc%i:2"%(self.num_rings), "0:add"]
        else: # top=add, btm=drop
            link1 += ["0:dc%i:3"%(self.num_rings), "0:add"]
            link2 += ["1:dc%i:2"%(self.num_rings), "0:drop"]

        self.link(*link1)
        self.link(*link2)


class RBS(pt.Component):
    num_ports = 4

    def __init__(
        self,
        tau_stacked= 0.5,
        eta_stacked=1 / 2,
        theta_t= np.pi/3,
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
        self.trainable = trainable

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
        T_B = generate_transfer(S_B)

        # Create 2x2 matrix SI/S2________________________________________________________________________
        E0 = self.eta
        E1 = self.gamma
        E2 = self.gamma
        E3 = self.eta

        S_I = np.array([[E0, E1], [E2, E3]])
        T_I = generate_transfer(S_I)

        # Create 2x2 matrix ST/S3_______________________________________________________________________
        # Have to extract float from trainable parameter
        if self.trainable:
            theta_Ft = self.theta_t.item()
        else:
            theta_Ft = self.theta_t

        E0 = self.tau * np.exp(-1j * theta_Ft)
        E1 = -np.conjugate(self.kappa)*np.exp(-1j * theta_Ft/2)
        E2 = self.kappa * np.exp(-1j * theta_Ft / 2)
        E3 = self.tau
        S_T = np.array([[E0, E1], [E2, E3]])
        T_T = generate_transfer(S_T)


        A = T_B.dot(T_I)
        S_DB = np.transpose(generate_scattering(A.dot(T_T)))


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


class APF(pt.Component):
    num_ports = 4

    def __init__(
        self,
        tau_ap= 0.5,
        theta_ap= 0,
        trainable=True,
        name=None,
    ):
        super(APF, self).__init__(name=name)

        self.trainable = trainable
        self.tau = np.sqrt(tau_ap)
        self.kappa = 1j * np.sqrt(1 - tau_ap)

        if not trainable:
            self.theta_ap = theta_ap
        else:
            theta_ap = torch.tensor(theta_ap, dtype=torch.float64)
            self.theta_ap = torch.nn.Parameter(data=theta_ap)

    def set_S(self, S):
        # Create 2x2 matrix SB/S1___________________________________________________________________________
        # if self.trainable:
        #     theta_AP = self.theta_ap.item()
        # else:
        #     theta_AP = self.theta_ap

        E0_Re = 1
        E0_Im = 0

        E1_Re = 0
        E1_Im = 0

        E2_Re = 0
        E2_Im = 0

        #E3 = np.conjugate(1 / (self.tau - (abs(self.kappa)**2) * np.exp(-1j * theta_AP) / (1 - self.tau * np.exp(-1j * self.theta_AP))))

#______________THESIS EQUATION UNITARY???_____________________________________________________________________________________________________
        #E3_Re = self.tau * ((np.imag(self.kappa))**2 + 1) + (self.tau**2 + 1) * np.imag(self.kappa) * torch.sin(self.theta_ap)
        #E3_Re = E3_Re / ((-self.tau * torch.cos(self.theta_ap))**2 + (np.imag(self.kappa) + self.tau * torch.sin(self.theta_ap))**2)
        #
        #E3_Im = (1 - self.tau**2) * (np.imag(self.kappa) * torch.cos(self.theta_ap))
        #E3_Im = E3_Im / ((-self.tau * torch.cos(self.theta_ap))**2 + (np.imag(self.kappa) + self.tau * torch.sin(self.theta_ap))**2)
# ______________LESS THAN 1 FOR SOME REASON???_____________________________________________________________________________________________________

        E3_Re = self.tau + self.tau * (self.tau**2 + np.imag(self.kappa)) - (self.tau**2 + np.imag(self.kappa)) * torch.cos(self.theta_ap) - self.tau**2 * torch.cos(self.theta_ap)
        E3_Re = E3_Re / ((self.tau - (self.tau**2 + np.imag(self.kappa)) * torch.cos(self.theta_ap))**2 + ((self.tau**2 + np.imag(self.kappa)) * torch.sin(self.theta_ap))**2)
        #
        E3_Im = -np.imag(self.kappa) * torch.sin(self.theta_ap)
        E3_Im = E3_Im / ((self.tau - (self.tau**2 + np.imag(self.kappa)) * torch.cos(self.theta_ap))**2 + ((self.tau**2 + np.imag(self.kappa)) * torch.sin(self.theta_ap))**2)
# ____________________________________________________________________________________________________________________________________________________________



        # S_DB = np.array([[E0, E1], [E2, E3]])


        S[0, 0, 0, 1] = S[0, 0, 1, 0] = E0_Re
        S[0, 0, 3, 1] = S[0, 0, 2, 0] = E2_Re
        S[0, 0, 0, 2] = S[0, 0, 1, 3] = E1_Re
        S[0, 0, 3, 2] = S[0, 0, 2, 3] = E3_Re

        S[1, 0, 0, 1] = S[1, 0, 1, 0] = E0_Im
        S[1, 0, 3, 1] = S[1, 0, 2, 0] = E2_Im
        S[1, 0, 0, 2] = S[1, 0, 1, 3] = E1_Im
        S[1, 0, 3, 2] = S[1, 0, 2, 3] = E3_Im




class RISQNET(pt.Network):
    def __init__(self, name=None):
        super(RISQNET, self).__init__(name=name)

        self.source = pt.Source()
        self.through = self.add = self.drop = pt.Detector()
        self.dc1 = RBS(theta_t=np.pi/3, theta_b=np.pi/3)
        self.dc2 = RBS(theta_t=np.pi/3, theta_b=np.pi/3)
        self.ap1 = APF(theta_ap=0)
        self.ap2 = APF(theta_ap=0)

        self.link('source:0', '0:dc1:1', '0:ap1:1', '0:dc2:1', '0:ap2:1', '0:through')
        self.link('add:0', '3:dc1:2', '3:ap1:2', '3:dc2:2', '3:ap2:2', '0:drop')

#______________________________________________MAIN________________________________________________________
# Define enviroment variables
device = 'cpu' # 'cpu' or 'cuda'
#RISQ = Crow(num_rings=2, random_parameters=True).to(device)
RISQ = RISQNET().to(device)

dt = 1e-14 #[s]
total_time = 2000*dt #[s]
time = np.arange(0, total_time, dt)

num_epochs = 30 # number of training cycles
learning_rate = 0.1 # multiplication factor for the gradients during optimization.
lossfunc = torch.nn.MSELoss()
optimizer = torch.optim.Adam(RISQ.parameters(), learning_rate)
for p in RISQ.parameters():
    print(p)




# Define environment
RISQ_TEST_ENVIRONMENT = pt.Environment(wavelength = 1e-6, freqdomain=False, time=time,)
RISQ_TRAIN_ENVIRONMENT = pt.Environment(wavelength = 1e-6, freqdomain=False, time=time, grad=True)

# Run Test
pt.set_environment(RISQ_TEST_ENVIRONMENT)

with RISQ_TRAIN_ENVIRONMENT:
    TEST_OUT = RISQ(source=1)
    RISQ.plot(TEST_OUT)
    plt.show()

total_power_out = TEST_OUT.data.cpu().numpy()[-1].sum()
target = np.ones(2) * total_power_out * 0.5
target = np.insert(target, 1, 0)

target = torch.tensor(target, device=RISQ.device, dtype=torch.get_default_dtype())


# loop over the training cycles:
with RISQ_TRAIN_ENVIRONMENT:
    for epoch in range(num_epochs):
        print(epoch)
        optimizer.zero_grad()
        detected = RISQ(source=1)[-1,0,:,0] # get the last timestep, the only wavelength, all detectors, the only batch
        loss = lossfunc(detected, target) # calculate the loss (error) between detected and target
        print(detected)
        print(target)
        loss.backward() # calculate the resulting gradients for all the parameters of the network
        optimizer.step() # update the networks parameters with the gradients
        del detected, loss # free up memory (important for GPU)
        if epoch % 10 == 0:
            detected = RISQ(source=1)  # get all timesteps, the only wavelength, all detectors, the only batch
            RISQ.plot(detected)
            plt.show()

for p in RISQ.parameters():
    print(p)

detected = RISQ(source=1)  # get all timesteps, the only wavelength, all detectors, the only batch
RISQ.plot(detected)
plt.show()