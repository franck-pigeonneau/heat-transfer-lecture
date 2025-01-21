#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 21:12:24 2020

Solve the Blasius equation to describe the self-similar solution of the
laminar boundary layer.

@author: Franck Pigeonneau CEMEF/CFL
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

def ydot(x,y):

    """

    Parameters
    ----------
    x : Float
        self-similar variable.
    y : array of float, 3 components.
        
        y[0] is the stream function
        y[1] its first derivative
        y[2] its second derivative
        
    Returns
    -------
    dydx : array of 3 components in float
    
        dydx[i]: first derivative of y[i]
        dydx[2] corresponds to the Blassius equation.

    """
    dydx=np.zeros(3)
    dydx[0]=y[1]
    dydx[1]=y[2]
    dydx[2]=-y[0]*y[2]*0.5
    return dydx
#end ydot(x,y)

def fzero(d2fdeta2,xmax):
    """
    
    Parameters
    ----------
    d2fdeta2 : Float
        Variable corresponding to the second derivative of the stream function.
        This variable must be adapted to find the first derivative of the stream
        function to be equal to 1 far from the wall.
        
    xmax : Float
        Maximum size of the self-similar variable.

    Returns
    -------
    Float
        Absolute difference between the first derivative of the stream function
        to 1 corresponding to the x-component of the velocity outside the
        boundary layer.

    """
    
    print(d2fdeta2[0])
    y0=np.array([0.,0.,d2fdeta2[0]])
    sol=solve_ivp(ydot,[0,xmax],y0,method='LSODA')
    
    # Return of the difference of the x-component of the velocity and 1
    return np.abs(sol.y[1,np.size(sol.y,1)-1]-1.)
#end fzero(d2fdeta2,xmax)

# Finding of the Blasius solution using a shoot method
# ----------------------------------------------------

d2fdeta2=0.2
xmax=10.
d2fdeta2=fsolve(fzero,d2fdeta2,args=(xmax,))
print('d2fdeta2=',d2fdeta2[0])

# Determination of the Blasius solution with the value of d2fdeta2
# ----------------------------------------------------------------
y0=np.array([0.,0.,d2fdeta2[0]])
sol=solve_ivp(ydot,[0,xmax],y0,method='LSODA')

# Plotting of the solution
# ------------------------

plt.figure()
plt.plot(sol.t,sol.y[1,:])
plt.xlabel(r'$y$')
plt.ylabel(r'$f^\prime$')

# Saving of the solution
# ----------------------

N=np.size(sol.t)
A=np.zeros((4,N))
for i in range(N):
    A[0,i]=sol.t[i]
    for k in range(3):
        A[k+1,i]=sol.y[k,i]
    #end for
#end for

np.savetxt('blasiussol.dat',np.transpose(A))



