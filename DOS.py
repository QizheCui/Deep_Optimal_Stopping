import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
This file contains the implementation of the Deep Optimal Stopping (DOS) algorithm as described
in the paper "Deep Optimal Stopping" by Sebastian Becker, Patrick Cheridito, Arnulf Jentzen
"""

#Parameters
T = 3.0    # final time
N = 9  # number of stopping times
dt = T/N  #time intervals
batch_size = 8192 #batch size

#Simulate Geometric brownian motion paths 
def GBM(d, mu, sigma, S0, T, dt, number_of_paths, seed=None):
    """
    Efficiently simulates number_of_paths many d-dimensional geometric brownian motion (GBM) sample paths.

    Arguments:
    d : int
        The dimension of the GBM to be simulated.
    mu : array
        Drift values in an array of shape (d,).
    sigma : array
        Volatilities in an array of shape (d,).
    S0 : array
        Initial values of the GBM in an array of shape (d,).
    T : float
        Specifies the time interval [0, T] in which the GBM will be simulated.
    dt : float
        Specifies the time increments.
    number_of_paths : int
        Number of sample paths to be simulated.
    seed : int, optional
        The seed for the random number generator to ensure reproducibility.

    Returns:
    A numpy array of GBM simulations of shape (number_of_paths, d, n) where n = T/dt, more memory efficient.
    """
    if seed is not None:
        np.random.seed(seed)  # Set the seed if provided

    n = int(T / dt)  # number of time steps
    dt_sqrt = np.sqrt(dt)
    
    # Precompute drift and diffusion terms
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_term = sigma * dt_sqrt
    
    # Initialize the simulations array
    S = np.empty((number_of_paths, d, n + 1), dtype=np.float32)
    S[:, :, 0] = S0

    # Simulate paths
    for t in range(1, n + 1):
        Z = np.random.randn(number_of_paths, d).astype(np.float32)
        S[:, :, t] = S[:, :, t-1] * np.exp(drift_term + diffusion_term * Z) # exact solution of GBM

    return S

#computes g values of a brownian motion of the form of output of above function
def g(x,r,k,dt):
  """
  Computes the discounted payoff of a European call option at time 0.
  Parameters:
    x : numpy array
        The simulated paths of the GBM.
    r : float
        The risk-free interest rate.
    k : float
        The strike price.
    dt : float
        The time increment.
    Returns:
    numpy array
        The discounted payoff of a European call option at time 0.
  """
  y = np.maximum(np.amax(x - k, axis = 1), 0) #max(S1,...,Sd) - k
  z = np.ones((x.shape[0], x.shape[2])) # x.shape[0] is number of paths, x.shape[2] is number of time steps
  z[:, 0] = np.zeros((x.shape[0])) #initialize z0 = 0
  z = -r*dt*np.cumsum(z, axis =1) 
  z = np.exp(z) # e^(-r*t), discount factor
  return y * z # g = (max(S1,...,Sd) - k)^(+) * e^(-r*t), discounted back to time 0

#Creates neural network
def create_model(d):
    """
    Creates a neural network with 2 hidden layers of 40+d units
    Includes batch norm layers
    """
    model = nn.Sequential(
    nn.Linear(d, d+40), # input layer
    nn.BatchNorm1d(40+d), # batch normalization
    nn.ReLU(), # activation function
    nn.Linear(d+40, d+40), 
    nn.BatchNorm1d(d+40),
    nn.ReLU(),
    nn.Linear(d+40, 1),
    nn.Sigmoid()
    )
    return model

#initiates dictionaries for f,F,l at maturity time N
#that will contain functions F (soft stopping decision),f (stopping decision) and l (stopping time) from the paper
def fN(x):
    return 1 # at maturity we have to stop
def FN(x):
    return 1.0 # at maturity we have to stop
def lN(x):    #can take input a vector of values
    """
    Argument:
    x: a tensor of shape (k,d,1) which contains Nth values of brownian paths for k samples
    Outputs:
    Stopping times as a tensor of shape (k, ). (in this case it will just output [N-1, N-1, ..., N-1])
    """
    ans = N  * np.ones(shape = (x.shape[0], ))
    ans = ans.astype(int)
    return ans

# dictionaries containing stopping times, stopping decisions and soft stopping decisions
l = {N: lN} # dictionary containing stopping times, initialized with lN
f = {N: fN} #dictionary containing hard stopping decisions, initialized with fN
F = {N: FN} #dictionary containing soft stopping decisions, initialized with FN

#initiates dictionaries for f,F,l at time i<N
def train(X, r, k, dt, model, i, optimizer, number_of_training_steps, batch_size):
  """
  Trains the model for the ith stopping time where i is between 0 and N-1
  Arguments:
  X: tensor of shape (3000+d, 8192, d, 10) containing paths as specified in the paper
  r: risk free rate
  k: strike price
  dt: time interval
  model: neural network model
  i: stopping time index
  optimizer: optimizer to be used for training
  number_of_training_steps: number of training steps
  batch_size: batch size
  """
  for j in range(number_of_training_steps):
    batch = X[j] #batch of paths
    batch_now = batch[:, :, i] # the ith stopping time values
    batch_gvalues = g(batch,r,k,dt) # discounted payoff values at ith stopping time
    batch_gvalues_now = batch_gvalues[:, i].reshape(1, batch_size) # reshaping to make it compatible with the model 
    batch = torch.from_numpy(batch).float().to(device) # storing the batch in the device, preferably GPU
    Z = batch_gvalues[range(batch_size), l[i+1](batch)].reshape(1, batch_size)
    batch_now = torch.from_numpy(batch_now).float().to(device) 
    batch_gvalues_now = torch.from_numpy(batch_gvalues_now).float().to(device) 
    Z = torch.from_numpy(Z).float().to(device) 

    #compute loss
    z = model(batch_now) # model output
    ans1 = torch.mm(batch_gvalues_now, z) # z is a column vector, torch.mm is matrix multiplication, this is the first term in the loss function
    ans2 = torch.mm( Z, 1.0 - z) # z is a column vector, torch.mm is matrix multiplication, this is the second term in the loss function
    loss = - 1 / batch_size * (ans1 + ans2) # loss function
    
    #apply updates
    optimizer.zero_grad() # zero the gradients
    loss.backward() # backpropagation
    optimizer.step() # update the weights

  print(f"the model for {i}th stopping time has been trained")


def fi(x, i, F):
    """
    the function that returns the stopping decision for ith stopping time
    Arguments:
    x: a tensor of shape (k, d) which contains ith values of brownian paths for k samples
    i: ith stopping time
    F: dictionary of models
    Outputs:
    hard Stopping decisions as a tensor of shape (k, ). (in this case it will just output 1 if x >= 1/2 else 0)
    """
    func = F[i].eval()
    return torch.ceil(func(x) - 1/2)

def li(x, i, f, l):
    """
    the function that returns the stopping time at ith stopping time
    Arguments:
    x: a tensor of shape (k, d) which contains ith values of brownian paths for k samples
    i: ith stopping time
    f: dictionary of stopping decision functions
    l: dictionary of stopping time functions
    Outputs:
    li Stopping times as a tensor of shape (k, ).
    """
    a = f[i](x[:,:,i]).cpu().detach().numpy().reshape(list(x[:,:,i].size())[0], )
    return ((i)*a + np.multiply(l[i+1](x), (1-a))).astype("int32")