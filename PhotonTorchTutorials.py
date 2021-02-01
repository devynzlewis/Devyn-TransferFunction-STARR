import torch
import numpy as np
import matplotlib.pyplot as plt
import photontorch as pt
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

dt = 1e-14 #[s]
total_time = 2000*dt #[s]
time = np.arange(0, total_time, dt)
c = 299792458 #[m/s]
ring_length = 50e-6 #[m]
transmission = 0.7 #[]
wavelengths = 1e-6*np.linspace(1.50, 1.6, 1000) #[m]

# create add-drop network using context manager

with pt.Network() as nw:
    nw.term_in = pt.Source()
    nw.term_pass = nw.term_add = nw.term_drop = pt.Detector()
    nw.dc1 = nw.dc2 = pt.DirectionalCoupler(1-transmission)
    nw.wg1 = nw.wg2 = pt.Waveguide(0.5*ring_length, loss=0, neff=2.86)
    nw.link('term_in:0', '0:dc1:2', '0:wg1:1', '1:dc2:3', '0:term_drop')
    nw.link('term_pass:0', '1:dc1:3', '0:wg2:1', '0:dc2:2', '0:term_add')

with pt.Environment(wl=np.mean(wavelengths), t=time):
    detected = nw(source=1)
    nw.plot(detected)
    plt.show()

