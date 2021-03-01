import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm # [pip install tqdm]
import torch # [conda install pytorch -c pytorch, only python 3!]
import photontorch as pt # [pip install photontorch] my simulation/optimization library

# c = 299792458.0 # speed of light
# ring_length = 50e-6 #[m]
# ng=3.4 # group index
# neff=2.34 # effective index



#_______________________________Try to define RISQ gate as one component_______________________________________


class RisqGate(pt.Component):
    """ RISQY BOI
    """

    # photontorch requires you to explicitly define the number of
    # ports in the component as a class variable:
    num_ports = 4

    def __init__(
        self,
        tau_stacked=1 / 2,
        eta_stacked = 1 / 2,
        theta_t = np.pi / 3,
        theta_b = np.pi / 3,
        tau_ap = 1 / 2,
        theta_ap = 0,
        trainable=True, # a flag to make the the component trainable or not
        name=None, # name of the waveguide
    ):
        """ creation of a new waveguide """
        super(RisqGate, self).__init__(name=name)# always initialize parent first
        # Handle inputs
        self.tau_stacked = float(tau_stacked)
        self.eta_stacked = float(eta_stacked)
        self.theta_t = float(theta_t)
        self.theta_b = float(theta_b)
        self.tau_ap = float(tau_ap)


        # handle phase input
        theta_ap = float(theta_ap) % (2*np.pi)
        if not trainable: # if the network is not trainable, just store it as a normal float:
            self.theta_ap = theta_ap
        else: # else, make an optimizable parameter out of it:
            # create a torch tensor from the phase
            theta_ap = torch.tensor(theta_ap, dtype=torch.float64)
            # store the phase as a optimizable parameter
            self.theta_ap = torch.nn.Parameter(data=theta_ap)

    # def set_delays(self, delays):
    #     """ set the delays for time-domain simulations """
    #     delays[:] = self.ng * self.length / self.env.c

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
        # Initialize Stuff___________________________________________________________________________

        sqr_eta = np.sqrt(self.eta_stacked)
        sqr_tau = np.sqrt(self.tau_stacked)
        kappa = 1j * np.sqrt(1 - self.tau_stacked)
        gamma = 1j * np.sqrt(1 - self.eta_stacked)

        # Create S_b matrix________________________________________________________________________

        t_b = sqr_tau
        s_b = -np.conjugate(kappa)*np.exp(-1j * self.theta_b/2)
        sPrime_b = kappa * np.exp(-1j * self.theta_b / 2)
        tPrime_b = sqr_tau*np.exp(-1j * self.theta_b)

        SB = np.array([[0, 0, 0, 0], [t_b, 0, 0, sPrime_b], [s_b, 0, 0, tPrime_b], [0, 0, 0, 0]])

        # Create S_t matrix________________________________________________________________________________

        t_t = sqr_tau * np.exp(-1j * self.theta_t)
        s_t = kappa * np.exp(-1j * self.theta_t) / 2
        sPrime_t = -np.conjugate(kappa) * np.exp(-1j * self.theta_t / 2)
        tPrime_t = sqr_tau

        ST = np.array([[0, 0, 0, 0], [t_t, 0, 0, sPrime_t], [s_t, 0, 0, tPrime_t], [0, 0, 0, 0]])

        # Create S_I Matrix_________________________________________________________________________________

        t_I = sqr_eta
        s_I = gamma
        sPrime_I = gamma
        tPrime_I = sqr_eta

        SI = np.array([[0, 0, 0, 0], [t_I, 0, 0, sPrime_I], [s_I, 0, 0, tPrime_I], [0, 0, 0, 0]])

        # Create S_DB Matrix _______________________________________________________________________________

        A = SB.dot(SI)
        S_DB = A.dot(ST)

        # Create All-Pass Matrix ___________________________________________________________________________

        a_c = np.conjugate(1 / (self.tau_ap - (abs(kappa) ** 2) * np.exp(-1j * self.theta_ap) / (1 - self.tau_ap * np.exp(-1j * self.theta_ap))))

        AP = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, a_c], [0, 0, 0, 0]])

        # Create the 4x4 RISQ Matrix ________________________________________________________________________

        A = S_DB.dot(AP)
        RISQ = A.dot(S_DB)

        # Add the RISQ matrix to the S_Matrix tensor__________________________________________________________

        S[0, :, 1, 0] = np.real(RISQ[1, 0])
        S[0, :, 2, 0] = np.real(RISQ[2, 0])
        S[0, :, 1, 3] = np.real(RISQ[1, 3])
        S[0, :, 2, 3] = np.real(RISQ[2, 3])

        S[1, :, 1, 0] = np.imag(RISQ[1, 0])
        S[1, :, 2, 0] = np.imag(RISQ[2, 0])
        S[1, :, 1, 3] = np.imag(RISQ[1, 3])
        S[1, :, 2, 3] = np.imag(RISQ[2, 3])



class RisqNet(pt.Network):
    def __init__(
        self,
        name=None
    ):
        super(RisqNet, self).__init__(name=name) # always initialize parent first

        # define subcomponents
        self.source = pt.Source()
        self.through = self.add = self.drop = pt.Detector()
        self.RISQ = RisqGate()

        # link subcomponents together:

        # The `link` method takes an arbitrary number of string arguments.
        # Each argument contains the component name together with a port numbe
        # in front of and a port number behind the name (e.g. `"0:wg:1"`).
        # The port number behind the name will connect to the port number
        # in front of the next name. The first component does not need
        # a port number in front of it, while the last component does
        # not need a port number behind.
        self.link('source:0', '0:RISQ:1', '0:through')
        self.link('add:0', '3:RISQ:2', '0:drop')


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

    def __init__(self, coupling=0.5, name=None):
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


class AllPass(pt.Network):
    def __init__(
        self,
        ring_length=1e-5, #[um] length of the ring
        ring_loss=1, #[dB]: roundtrip loss in the ring
        name=None
    ):
        super(AllPass, self).__init__(name=name) # always initialize parent first

        # handle arguments:
        self.ring_length = float(ring_length)
        self.ring_loss = float(ring_loss),

        # define subcomponents
        self.source = pt.Source()
        self.detector = pt.Detector()
        self.dc = DirectionalCoupler()
        self.wg = Waveguide(length=ring_length, loss=ring_loss/ring_length)

        # link subcomponents together:

        # The `link` method takes an arbitrary number of string arguments.
        # Each argument contains the component name together with a port numbe
        # in front of and a port number behind the name (e.g. `"0:wg:1"`).
        # The port number behind the name will connect to the port number
        # in front of the next name. The first component does not need
        # a port number in front of it, while the last component does
        # not need a port number behind.

        self.link('source:0', '0:dc:2', '0:wg:1', '3:dc:1', '0:detector')


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


def plot_detected(num_rings=5):
    crow = Crow(num_rings=num_rings)
    detected = crow(source=1)
    crow.plot(detected)
    plt.show()


# # Time Domain 2 ring crow filter____________________________________________________________________________________
# device = 'cpu' # 'cpu' or 'cuda'
# crow = Crow(num_rings=2, random_parameters=True).to(device)
#
# neff = np.sqrt(12.1)
# wl = 1.55e-6
# dt = 0.5e-14
# total_time = 2e-12
# time = np.arange(0,total_time,dt)
#
# num_epochs = 100 # number of training cycles
# learning_rate = 0.1 # multiplication factor for the gradients during optimization.
# lossfunc = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(crow.parameters(), learning_rate)
# print(crow.parameters())
# # define the simulation environment:
# env = pt.Environment(
#     wavelength = 1.5e-6,
#     freqdomain=False, # we will be doing frequency domain simulations
# )
# # set the global simulation environment:
# pt.set_environment(env)
#
# train_env = pt.Environment(
#     wavelength = 1e-6, #[m]
#     time=time,
#     freqdomain=False, # we will be doing frequency domain simulations
#     grad=True, # we need to enable gradients to be able to optimize
# )
#
# # over the trainin domain:
# with train_env:
#     detected = crow(source=1)[:,0,:,0] # single timestep, all wls, (drop detector=0; through detector=2), single batch
#     crow.plot(detected)
#     plt.show()
#
# total_power_out = detected.data.cpu().numpy()[-1].sum()
# target = np.ones(2)*total_power_out/2
# target = np.insert(target, 0, 0)
#
# target = torch.tensor(target, device=crow.device, dtype=torch.get_default_dtype())
#
# # loop over the training cycles:
# with train_env:
#     for epoch in range(num_epochs):
#         print(epoch)
#         optimizer.zero_grad()
#         detected = crow(source=1)[-1,0,:,0] # get the last timestep, the only wavelength, all detectors, the only batch
#         loss = lossfunc(detected, target) # calculate the loss (error) between detected and target
#         print(detected)
#         print(target)
#         loss.backward() # calculate the resulting gradients for all the parameters of the network
#         optimizer.step() # update the networks parameters with the gradients
#         del detected, loss # free up memory (important for GPU)
#         if epoch % 10 == 0:
#             detected = crow(source=1)  # get all timesteps, the only wavelength, all detectors, the only batch
#             crow.plot(detected)
#             plt.show()
#
#
#
#
#
# with train_env:
#     detected = crow(source=1) # get all timesteps, the only wavelength, all detectors, the only batch
#     crow.plot(detected)
#     plt.show()
#
dt = 0.5e-14
total_time = 2e-12
time = np.arange(0,total_time,dt)


risq = RisqGate()

env = pt.Environment(
    wavelength = 1e-6, #[m]
    freqdomain=False, # we will be doing frequency domain simulations
    time=time
)

pt.set_environment(env)

detected = risq(source=1)

risq.plot(detected)
plt.show()



