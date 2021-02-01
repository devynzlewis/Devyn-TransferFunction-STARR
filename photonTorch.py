import torch
import numpy as np
import matplotlib.pyplot as plt
import photontorch as pt


class AllPass(pt.Network):
    def __init__(self, length=1e-5, neff=2.34, ng=3.4, loss=1000):
        super(AllPass, self).__init__() # always initialize first
        self.wg1 = pt.Waveguide(length=length, neff=neff, ng=ng, loss=loss, trainable=True)
        self.dc1 = pt.DirectionalCoupler(coupling=0.3, trainable=True)
        self.link("dc1:2", "0:wg1:1", "3:dc1")

# see if the network is terminated
#print(torch.where(AllPass().free_ports_at)[0])


class Circuit(pt.Network):
    def __init__(self, length=1e-5, neff=2.34, ng=3.4, loss=1000):
        super(Circuit, self).__init__()
        self.allpass = AllPass(length, neff, ng, loss)
        self.source = pt.Source()
        self.detector = pt.Detector()

        # note that you link with the allpass circuit as if it was
        # a single component. You do not link with the subcomponents
        # of the allpass component!
        self.link("source:0", "0:allpass:1", "0:detector")

# see if the network is terminated
#print(torch.where(Circuit().free_ports_at)[0])

circuit = Circuit(length=1.2e-5, neff=2.84, ng=3.2, loss=3e4)

# create simulation environment
freq_env = pt.Environment(
    wl=1e-6*np.linspace(1.45, 1.65, 1000),
    freqdomain=True
)



for p in circuit.parameters():
    print(p)


# define simualation environment
train_env = pt.Environment(
    wl=1525e-9, # we want minimal transmission at this wavelength
    freqdomain=True, # we will do frequency domain simulations
    grad=True, # gradient need to be tracked in order to use gradient descent.
)

# define target for the simulation, lets take 0 (ideally no transmission at resonance)
target = 0

# let's define an optimizer.
# The Adam optimizer is generally considered to be the best
# gradient descent optimizer out there:
optimizer = torch.optim.Adam(circuit.parameters(), lr=0.1)

# do the training
with train_env:
    for epoch in range(100):
        optimizer.zero_grad() # reset the optimizer and gradients
        detected = circuit(source=1) # simulate
        loss = ((detected-target)**2).mean() # calculate mse loss
        loss.backward() # calculate the gradients on the parameters
        optimizer.step() # update the parameters according to the gradients

# view result
with freq_env:
    detected = circuit(source=1) # constant source with amplitude 1
    circuit.plot(detected);
    plt.show()

# print parameters
for p in circuit.parameters():
    print(p)


