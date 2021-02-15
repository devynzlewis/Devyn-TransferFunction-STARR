import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm # [pip install tqdm]
import torch # [conda install pytorch -c pytorch, only python 3!]
import photontorch as pt # [pip install photontorch] my simulation/optimization library




#
# # ___________________T1 Start______________________________________________________________________________________
# class AllPass(pt.Network):
#     def __init__(self, length=1e-5, neff=2.34, ng=3.4, loss=1000):
#         super(AllPass, self).__init__() # always initialize first
#         self.wg1 = pt.Waveguide(length=length, neff=neff, ng=ng, loss=loss, trainable=True)
#         self.dc1 = pt.DirectionalCoupler(coupling=0.3, trainable=True)
#         self.link("dc1:2", "0:wg1:1", "3:dc1")
#
# # see if the network is terminated
# #print(torch.where(AllPass().free_ports_at)[0])
#
#
# class Circuit(pt.Network):
#     def __init__(self, length=1e-5, neff=2.34, ng=3.4, loss=1000):
#         super(Circuit, self).__init__()
#         self.allpass = AllPass(length, neff, ng, loss)
#         self.source = pt.Source()
#         self.detector = pt.Detector()
#
#         # note that you link with the allpass circuit as if it was
#         # a single component. You do not link with the subcomponents
#         # of the allpass component!
#         self.link("source:0", "0:allpass:1", "0:detector")
#
# # see if the network is terminated
# #print(torch.where(Circuit().free_ports_at)[0])
#
# circuit = Circuit(length=1.2e-5, neff=2.84, ng=3.2, loss=3e4)
#
# # create simulation environment
# freq_env = pt.Environment(
#     wl=1e-6*np.linspace(1.45, 1.65, 1000),
#     freqdomain=True
# )
#
#
#
# #for p in circuit.parameters():
#     #print(p)
#
#
# # define simualation environment
# train_env = pt.Environment(
#     wl=1525e-9, # we want minimal transmission at this wavelength
#     freqdomain=True, # we will do frequency domain simulations
#     grad=True, # gradient need to be tracked in order to use gradient descent.
# )
#
# # define target for the simulation, lets take 0 (ideally no transmission at resonance)
# target = 0
#
# # let's define an optimizer.
# # The Adam optimizer is generally considered to be the best
# # gradient descent optimizer out there:
# optimizer = torch.optim.Adam(circuit.parameters(), lr=0.1)
#
# # do the training
# with train_env:
#     for epoch in range(100):
#         optimizer.zero_grad() # reset the optimizer and gradients
#         detected = circuit(source=1) # simulate
#         loss = ((detected-target)**2).mean() # calculate mse loss
#         loss.backward() # calculate the gradients on the parameters
#         optimizer.step() # update the parameters according to the gradients
#
# # view result
# with freq_env:
#     detected = circuit(source=1) # constant source with amplitude 1
#     circuit.plot(detected);
#     #plt.show()
#
# # print parameters
# #for p in circuit.parameters():
#     #print(p)

#_______________________T1 End / T2 Start_______________________________________________

# Power vs frequency plot of an all-pass resonator

# First part uses analytical formula, next part uses photontorch.

# dt = 1e-14 # Timestep of the simulation
# total_time = 2.5e-12 # Total time to simulate
# time = np.arange(0, total_time, dt) # Total time array
# loss = 1 # [dB] (alpha) roundtrip loss in ring
# neff = 2.34 # Effective index of the waveguides
# ng = 3.4
# ring_length = 1e-5 #[m] Length of the ring
# transmission = 0.5 #[] transmission of directional coupler
# wavelengths = 1e-6*np.linspace(1.5,1.6,1000) #[m] Wavelengths to sweep over
#
#
# def frequency():
#     #''' Analytic Frequency Domain Response '''
#     detected = np.zeros_like(wavelengths)
#     for i, wl in enumerate(wavelengths):
#         wl0 = 1.55e-6
#         neff_wl = neff + (wl0-wl)*(ng-neff)/wl0 # we expect a linear behavior with respect to wavelength
#         out = np.sqrt(transmission) - 10**(-loss/20.)*np.exp(2j*np.pi*neff_wl*ring_length/wl)
#         out /= (1 - np.sqrt(transmission)*10**(-loss/20.)*np.exp(2j*np.pi*neff_wl*ring_length/wl))
#         detected[i] = abs(out)**2
#     return detected
#
#
# def plot_frequency(detected, **kwargs):
#     #''' Plot detected power vs time '''
#     labels = kwargs.pop('labels', ['through','drop','add'])
#     plots = plt.plot(wavelengths*1e9, detected, **kwargs)
#     plt.xlabel('Wavelengths [nm]')
#     plt.ylabel('Transmission')
#     if labels is not None: plt.figlegend(plots, labels, loc='upper center', ncol=len(labels)%5)
#     plt.show()
#
#
# detected_target = frequency()
# plot_frequency(detected_target)
#
# class AllPass(pt.Network):
#     def __init__(self):
#         super(AllPass, self).__init__()
#         self.src = pt.Source()
#         self.wg_in = pt.Waveguide(0.5*ring_length, neff=neff, ng=ng)
#         self.dc = pt.DirectionalCoupler(1-transmission)
#         self.wg_through = pt.Waveguide(0.5*ring_length, neff=neff, ng=ng)
#         self.wg_ring = pt.Waveguide(ring_length, loss=loss/ring_length, neff=neff)
#         self.det = pt.Detector()
#         self.link('src:0', '0:wg_in:1', '0:dc:1', '0:wg_through:1', '0:det')
#         self.link('dc:2', '0:wg_ring:1', '3:dc')
#
# nw = AllPass()
#
# # Next we create the environment which contains global information like wavelength and timesteps
#
# environment = pt.Environment(wl=np.mean(wavelengths), t=time)
#
# pt.set_environment(environment)
#
# # detected = nw(source=1)
#
# with pt.Environment(wl=wavelengths, freqdomain=True) as env:
#     detected = nw(source=1)
#     print("This was detected inside the context manager:\n"
#           "We see an exact copy of the analytically predicted response, as is to be expected")
#     nw.plot(detected, label="simulation")
#     plt.plot(env.wavelength*1e9, detected_target, linestyle="dotted", linewidth=3, label="analytical")
#     plt.legend()
#     plt.show()
#
# print("This was detected outside the context manager, "
#       "with the default environment:")
# detected = nw(source=1)
# nw.plot(detected)
# plt.show()
#
# with pt.Environment(wl=wavelengths[::100], t=time):
#     detected = nw(source=1)
#     nw.plot(detected)
#     plt.show()


#_______________T2 End / T3 Start____________________________________________________________________________________

# dt = 1e-14 #[s]
# total_time = 2000*dt #[s]
# time = np.arange(0, total_time, dt)
# c = 299792458 #[m/s]
# ring_length = 50e-6 #[m]
# transmission = 0.7 #[]
# wavelengths = 1e-6*np.linspace(1.50, 1.6, 1000) #[m]
#
# # create add-drop network using context manager
#
# with pt.Network() as nw:
#     nw.term_in = pt.Source()
#     nw.term_pass = nw.term_add = nw.term_drop = pt.Detector()
#     nw.dc1 = nw.dc2 = pt.DirectionalCoupler(1-transmission)
#     nw.wg1 = nw.wg2 = pt.Waveguide(0.5*ring_length, loss=0, neff=2.86)
#     nw.link('term_in:0', '0:dc1:2', '0:wg1:1', '1:dc2:3', '0:term_drop')
#     nw.link('term_pass:0', '1:dc1:3', '0:wg2:1', '0:dc2:2', '0:term_add')
#
# with pt.Environment(wl=np.mean(wavelengths), t=time):
#     detected = nw(source=1)
#     nw.plot(detected)
#     plt.show()
#
# with pt.Environment(wl=wavelengths, freqdomain=True):
#     detected = nw(source=1)
#     nw.plot(detected)
#     plt.show()

#__________________________Rest of T3 is similar with grated couplers T4 Next_____________________________________________________

#__________________________We're creating an interferometer______________________________________________________________
# neff = np.sqrt(12.1)
# wl = 1.55e-6
# dt = 0.5e-9
# total_time = 2e-6
# time = np.arange(0,total_time,dt)
#
# class MichelsonCavity(pt.Network):
#     def __init__(self):
#         super(MichelsonCavity, self).__init__()
#         self.west = pt.Source()
#         self.north = self.east = self.south = pt.Detector()
#         self.m_west = pt.Mirror(R=0.9)
#         self.m_north = pt.Mirror(R=0.9)
#         self.m_east = pt.Mirror(R=0.9)
#         self.m_south = pt.Mirror(R=0.9)
#         self.wg_west = pt.Waveguide(0.43, neff=neff, trainable=False)
#         self.wg_north = pt.Waveguide(0.60, neff=neff, trainable=False)
#         self.wg_east = pt.Waveguide(0.95, neff=neff, trainable=False)
#         self.wg_south = pt.Waveguide(1.12, neff=neff, trainable=False)
#         self.dc = pt.DirectionalCoupler(coupling=0.5, trainable=False)
#         self.link('west:0','0:m_west:1', '0:wg_west:1', '0:dc:2', '0:wg_east:1', '0:m_east:1', '0:east')
#         self.link('north:0', '0:m_north:1', '0:wg_north:1', '1:dc:3', '0:wg_south:1', '0:m_south:1', '0:south')
#
# # create network
# nw = MichelsonCavity()
#
# # print out the parameters of the network:
# for p in nw.parameters():
#     print(p)
#
# with pt.Environment(wl=wl, t=time):
#     detected = nw(source=1)[:,0,:,0] # get all timesteps, the only wavelength, all detectors, the only batch
#
# nw.plot(detected)
# plt.show()
#
# num_epochs = 10 # number of training cycles
# learning_rate = 0.2 # multiplication factor for the gradients during optimization.
# lossfunc = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(nw.parameters(), learning_rate)
#
# total_power_out = detected.data.cpu().numpy()[-1].sum()
# target = np.ones(3)*total_power_out/3
# # The target should be a torch variable.
# # You can create a new torch variable with the right type and cuda type, straight from the network itself:
# target = torch.tensor(target, device=nw.device, dtype=torch.get_default_dtype())
#
#
# # loop over the training cycles:
# with pt.Environment(wl=wl, t=time, grad=True):
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         detected = nw(source=1)[-1,0,:,0] # get the last timestep, the only wavelength, all detectors, the only batch
#         loss = lossfunc(detected, target) # calculate the loss (error) between detected and target
#         loss.backward() # calculate the resulting gradients for all the parameters of the network
#         optimizer.step() # update the networks parameters with the gradients
#         del detected, loss # free up memory (important for GPU)
#         print(epoch)
#
#
# with pt.Environment(wl=wl, t=time):
#     detected = nw(source=1) # get all timesteps, the only wavelength, all detectors, the only batch
#     nw.plot(detected)
#     plt.show()

#_______________________________________________2x2 unitary_________________________________________________

#Frequency domain

# DEVICE = 'cpu'
# np.random.seed(0)
# torch.manual_seed(0)
# np.set_printoptions(precision=2, suppress=True)
# env = pt.Environment(freqdomain=True, num_t=1, grad=True)
# pt.set_environment(env)
# print(pt.current_environment())
#
#
# def array(tensor):
#     arr = tensor.data.cpu().numpy()
#     if arr.shape[0] == 2:
#         arr = arr[0] + 1j * arr[1]
#     return arr
#
#
# def tensor(array):
#     if array.dtype == np.complex64 or array.dtype == np.complex128:
#         array = np.stack([np.real(array), np.imag(array)])
#     return torch.tensor(array, dtype=torch.get_default_dtype(), device=DEVICE)
#
#
# def rand_phase():
#     return float(2*np.pi*np.random.rand())
#
#
# class Network(pt.Network):
#     def _handle_source(self, matrix, **kwargs):
#         expanded_matrix = matrix[:,None,None,:,:]
#         a,b,c,d,e = expanded_matrix.shape
#         expanded_matrix = torch.cat([
#             expanded_matrix,
#             torch.zeros((a,b,c,self.num_mc-d,e), device=expanded_matrix.device),
#         ], -2)
#         return expanded_matrix
#
#     def forward(self, matrix):
#         ''' matrix shape = (2, num_sources, num_sources)'''
#         result = super(Network, self).forward(matrix, power=False)
#         return result[:,0,0,:,:]
#
#     def count_params(self):
#         num_params = 0
#         for p in self.parameters():
#             num_params += int(np.prod(p.shape))
#         return num_params
#
#
# def unitary_matrix(m,n):
#     real_part = np.random.rand(m,n)
#     imag_part = np.random.rand(m,n)
#     complex_matrix = real_part + 1j*imag_part
#     if m >= n:
#         unitary_matrix, _, _ = np.linalg.svd(complex_matrix, full_matrices = False)
#     else:
#         _, _, unitary_matrix = np.linalg.svd(complex_matrix, full_matrices = False)
#     return unitary_matrix


# ___________________________________________________________________

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


# define target transmission

target = torch.tensor(0.0)

# Loss is simply mean squared error between Intensity and target Intensity
lossfunc = torch.nn.MSELoss()

# to define an optimizer, one needs to provide the parameters to optimize
# and the learning rate of the optimizer.
# the learning rate is an important parameter that needs to be tuned manually to
# the right value. A too large learning rate will result in an optimizer that cannot find the loss
# a too small value may result in a very long optimization time:
optimizer = torch.optim.Adam([allpass.wg.phase], lr=0.03) # let's just optimize the phase of the ring

train_env = pt.Environment(
    wl=1.55e-6, #[m]
    freqdomain=True, # we will be doing frequency domain simulations
    grad=True, # allow gradient tracking
)

with train_env: # temporarily override the global environment
    # we train for 400 training steps
    for i in range(400):
        optimizer.zero_grad() # set all the gradients to zero
        result = torch.squeeze(allpass(source=1)) # squeeze: 4D -> 0D
        loss = lossfunc(result, target) # MSE loss
        loss.backward() # calculate the gradients
        optimizer.step() # use the calculated gradients to perform an optimization step

print("loss: %.5f"%loss.item())

detected = allpass(source=1)
allpass.plot(detected)
plt.plot([1550,1550],[0,1])
plt.ylim(0,1)
plt.show()

# let's just optimize both the phase and the coupling
optimizer = torch.optim.Adam([allpass.wg.phase, allpass.dc.parameter], lr=0.03)

with train_env: # temporarily override the global environment
    # we train for 400 training steps
    for i in range(400):
        optimizer.zero_grad() # set all the gradients to zero
        result = torch.squeeze(allpass(source=1)) # squeeze: 4D -> 0D
        loss = lossfunc(result, target) # MSE loss
        loss.backward() # calculate the gradients
        optimizer.step() # use the calculated gradients to perform an optimization step

print("loss: %.5f"%loss.item())



detected = allpass(source=1)
allpass.plot(detected)
plt.plot([1550,1550],[0,1])
plt.ylim(0,1)
plt.show()

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


crow = Crow(num_rings=1)
detected = crow(source=1)
crow.plot(detected)
plt.show()

def plot_detected(num_rings=5):
    crow = Crow(num_rings=num_rings)
    detected = crow(source=1)
    crow.plot(detected)
    plt.show()

device = 'cpu' # 'cpu' or 'cuda'
crow = Crow(num_rings=10)

train_env = pt.Environment(
    wavelength = 1e-6*np.linspace(1.53, 1.58, 1001), #[m]
    freqdomain=True, # we will be doing frequency domain simulations
    grad=True, # we need to enable gradients to be able to optimize
)

# over the original domain
detected = crow(source=1)[0,:,2,0] # single timestep, all wls, (drop detector=0; through detector=2), single batch
crow.plot(detected)
plt.show()

# over the trainin domain:
with train_env:
    detected = crow(source=1)[0,:,2,0] # single timestep, all wls, (drop detector=0; through detector=2), single batch
    crow.plot(detected)
    plt.show()


target = np.zeros_like(train_env.wavelength)
target[train_env.wavelength > 1.55e-6] = 1
target[train_env.wavelength > 1.56e-6] = 0
target = torch.tensor(target, dtype=torch.get_default_dtype(), device=device)
crow.plot(target)
plt.show()

optimizer = torch.optim.Adam(crow.parameters(), lr=0.01)

range_ = range(5) # we train for 150 training steps
with train_env: # temporarily override the global environment
    for i in range_:
        print(i)
        crow.initialize()
        optimizer.zero_grad() # set all the gradients to zero
        result = crow(source=1)[0,:,0,0] # single timestep, all wls, (drop detector=0; through detector=2), single batch
        loss = lossfunc(result, target) # MSE loss
        loss.backward() # calculate the gradients
        optimizer.step() # use the calculated gradients to perform an optimization step
        # range_.set_postfix(loss=loss.item())

print("loss: %.5f"%loss.item())


# over the training domain:
with train_env:
    detected = crow(source=1)
    crow.plot(target, label="target")
    crow.plot(detected[:,:,:,:])
    plt.show()

