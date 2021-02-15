import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm # [pip install tqdm]
import torch # [conda install pytorch -c pytorch, only python 3!]
import photontorch as pt # [pip install photontorch] my simulation/optimization library


c = 299792458.0 # speed of light
ring_length = 50e-6 #[m]
ng=3.4 # group index
neff=2.34 # effective index

# define the simulation environment:
env = pt.Environment(
    wavelength = 1e-6*np.linspace(1.50, 1.6, 1001), #[m]
    freqdomain=True, # we will be doing frequency domain simulations
)

# set the global simulation environment:
pt.set_environment(env)

# one can always get the current environment from photontorch:
# print(pt.current_environment())

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

allpass = AllPass()

detected = allpass(source=1)
print(detected.shape)

allpass.plot(detected)
plt.show()