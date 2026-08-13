
import math
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import cm

L = 512
n0 = 1024
dx = L/n0
J = 1/dx**2
eta = 1
dt = 0.1
theta = 1e-8
W = math.sqrt(2 * eta * (theta / dt))
tauQ=128
nt=3201
Nsamples=10 # Number of samples generated; set to 3000 to generate 3000 samples.



def T(n0):
  h1=np.zeros((n0,n0))
  for i in range(1,n0): 
    h1[i,i-1]=J
    h1[i,i-n0+1]=J
    h1[i-1,i]=J
    h1[i-n0+1,i]=J
  for i in range(n0): 
    h1[i,i]=-2*J
  return h1

def retList():
    list = []
    for i in range(n0):
        list.append(0)
    return list

def eps(t):
  return t/tauQ

#The Langevin equation is numerically solved using the fourth-order Runge-Kutta method

for j in range(1, Nsamples+1):
  phi0 = retList()
  pi0 = retList()

  Noise=W*(np.random.normal(0,1,size=(nt,n0)))

  phi = np.zeros((nt+1,len(phi0)))
  pi = np.zeros((nt+1,len(pi0)))
  phi[0] = phi0
  pi[0] = pi0

  for i in range(nt):
    k1phi = pi[i]
    k1pi = -np.dot(eta,pi[i])+np.matmul(T(n0),phi[i])-(np.power(phi[i],3)-np.dot(eps(i*dt-1.5*tauQ),phi[i]))/2 +Noise[i]

    phi2 = phi[i] + np.dot(dt,k1phi/2)
    pi2 = pi[i] + np.dot(dt,k1pi/2)
    k2phi = pi2
    k2pi = -np.dot(eta,pi2)+np.matmul(T(n0),phi2)-(np.power(phi2,3)-np.dot(eps((i+1/2)*dt-1.5*tauQ),phi2))/2 +Noise[i]

    phi3 = phi[i] + np.dot(dt,k2phi/2)
    pi3 = pi[i] + np.dot(dt,k2pi/2)
    k3phi = pi3
    k3pi = -np.dot(eta,pi3)+np.matmul(T(n0),phi3)-(np.power(phi3,3)-np.dot(eps((i+1/2)*dt-1.5*tauQ),phi3))/2 +Noise[i]

    phi4 = phi[i] + np.dot(dt,k3phi)
    pi4 = pi[i] + np.dot(dt,k3pi)
    k4phi = pi4
    k4pi = -np.dot(eta,pi4)+np.matmul(T(n0),phi4)-(np.power(phi4,3)-np.dot(eps((i+1)*dt-1.5*tauQ),phi4))/2 +Noise[i]

    pi[i+1] = pi[i] + np.dot(dt,(k1pi +np.dot(2,k2pi)+ np.dot(2,k3pi) +k4pi)/6)
    phi[i+1] = phi[i] + np.dot(dt,(k1phi +np.dot(2,k2phi)+ np.dot(2,k3phi) +k4phi)/6)


  Phi = []
  # Save the short time-series input
  for i in range(2370, 2471,10):
    Phi.append(phi[i])
  # Save the final defect configuration
  Phi.append(phi[3200])

  with open(f"j{j}.txt", "w") as file:
    for sublist in Phi:
        file.write(f"[{','.join(map(str, sublist))}],")
    file.write("\n")  



           



                  

