# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 19:04:17 2020

@author: Andrea Bassi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


#if torch.cuda.is_available():
#    device = torch.device("cpu")
#else:
device = torch.device("cpu")
print('\n Using device:', device, '\n')    

# Create a 2D gaussian function with noise
# x,y are the cohordinates
x_lin = y_lin = np.arange(-1, 1.1, 0.1)
X_np, Y_np = np.meshgrid(x_lin,y_lin)
X0 = 0
Y0 = 0
w = 0.3 # waist

Z_np = 0.5 * np.exp((- (X_np-X0)**2 - (Y_np-Y0)**2)/w**2)
Z_np += 0.3 + 0.1* np.random.poisson(lam=1.0, size=Z_np.shape)

x = torch.from_numpy(X_np).float().to(device = device)
y = torch.from_numpy(Y_np).float().to(device = device)
z = torch.from_numpy(Z_np).float().to(device = device) 

# Define a net. 
# Here I don't really exploit the layers of Pytorch,
# I just define some parameters self.w 
# that are optimized using the gaussian forward 
class Net(torch.nn.Module):

    def __init__(self, w_in):
        super(Net,self).__init__()
        self.w = torch.nn.Parameter(w_in.clone()).to(device = device)
        self.w.requires_grad = True
        self.relu = torch.nn.ReLU()
        
    def forward(self, X, Y):
        amp = self.w[0]
        waist = self.w[1]
        bias = self.w[2]
        X0 = 0
        Y0 = 0
        z_pred = ( bias + amp * torch.exp((- (X-X0)**2 - (Y-Y0)**2)/waist**2)
                  ).to(device = device)
        return self.relu(z_pred) 
    
    def show_net_values(self, title = ''):
        print('\n', title, 'values:'
              '\n --> Amplitude =', self.w[0].detach().cpu().numpy(),
              '\n --> Waist =', self.w[1].detach().cpu().numpy(),
              '\n --> Bias =', self.w[2].detach().cpu().numpy(), 
              '\n'
              )


amplitude_guess = np.amax(Z_np)
waist_guess = np.std(Z_np)
bkg_guess =  np.mean(Z_np)
 
initial_guess = torch.tensor([amplitude_guess,
                              waist_guess,
                              bkg_guess
                              ]).to(device = device)

#initial_guess = torch.rand([3]).to(device = device)

net = Net(initial_guess)

net.show_net_values('Initial guess')

loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=0.1)


for t in range(50):
    
    # Forward pass: compute predicted y by passing x to the model.
    # This is equivalent to net.forward(x,y)
    z_pred = net(x,y)

    # Compute and print loss.
    loss = loss_fn(z_pred, z)
    if t % 5 == 4:
        
        print('step:',t,
              ',loss:', loss.item(),
             )

    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

net.show_net_values('Fitted')    

def show_image(data, title=''):
    plt.figure(figsize=(6, 6))
    plt.gray()
    plt.imshow(data,extent=[-1,1, -1,1])
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

input_data = z.detach().cpu().numpy()
output_data = z_pred.detach().cpu().numpy()

show_image(input_data,'Original')
show_image(output_data,'Predicted')


#plt.figure(figsize=(6, 6))
#
_fig, ax = plt.subplots()
center_idx = int(input_data.shape[0]/2)
ax.plot(x_lin, input_data[center_idx,:], 'gx', label = ('Original'))
ax.plot(x_lin, output_data[center_idx,:], 'r-', label = ('Predicted'))
ax.legend(loc='upper left', frameon=False)
ax.grid()
#ax.axis('equal')
ax.set(xlabel = 'x',
       ylabel = 'z',
       xlim = (-1, 1),
       title = 'Predicted vs Original'
       )
