#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Jan 24, 2023

Numerical computation of the natural convection along a heated vertical wall.

The sefl-similar problem depending on 2 unknows is written as a Cauchy problem 
of 5 unknowns.

@author: Franck Pigeonneau, CEMEF/CFL

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
    y : array of float, 4 components.
        
        y[0] is the stream function
        y[1] its first derivative
        y[2] its second derivative
        y[3] is the temperature
        y[4] is the first derivative of the temperature
        
    Returns
    -------
    dydx : array of 3 components in float
    
        dydx[i]: first derivative of y[i], for i=0 to 1
        dydx[2]: corresponds to the Blassius equation.
        dydx[3]: first derivative of the temperature
        dydx[4]: second derivative of the temperature

    """
    
    # Prandtl number
    Pr=0.73
    
    dydx=np.zeros(5)
    
    # Equation on the stream function
    dydx[0]=y[1]
    dydx[1]=y[2]
    dydx[2]=0.75*y[0]*y[2]-0.5*y[1]**2+y[3]
    
    # Equation on the temperature
    dydx[3]=y[4]
    dydx[4]=0.75*Pr*y[0]*y[4]
    
    # Return to the array of the derivatives
    return dydx
#end ydot(x,y)

def fshooting(derivate,xmax):
    """
    
    Parameters
    ----------
    d2fdeta2 : Float
        Variable corresponding to the second derivative of the stream function.
        This variable must be adapted to find the first derivative of the stream
        function to be equal to 0 far from the wall.
    dthetadeta : Float
        The first derivative of the temperature to be find in order to have,
        the temperature equal to 0 far from the wall.
    Pr: Float
        Prandtl number
    xmax : Float
        Maximum size of the self-similar variable.

    Returns
    -------
    Float
        Square root of the sum of y[1]**2 and y[3]**2 in xmax.

    """
    
    # Setting of the boundary condition in eta=0
    y0=np.array([0.,0.,derivate[0],1.,derivate[1]])
    
    # Compuation of the Cauchy problem
    sol=solve_ivp(ydot,[0,xmax],y0,method='LSODA')
    
    # Determination of the quadratic difference of y[1] et y[3] in eta=xmax 
    normyxmax=np.zeros(2)
    normyxmax[0]=np.abs(sol.y[1,np.size(sol.y,1)-1])
    normyxmax[1]=np.abs(sol.y[3,np.size(sol.y,1)-1])
    
    # Return to the value of normy1y3
    return normyxmax
#end fshooting(d2fdeta2,dthetadeta,Pr,xmax)

# Set of the full extinction of the streching domain
xmax=50.

# Finding of the solution using a shoot method
# --------------------------------------------
derivate=np.array([-0.5,-0.4])
derivate=fsolve(fshooting,derivate,args=(xmax,))
print('d2fdeta2=',derivate[0])
print('dthetadeta=',derivate[1])

# Determination of the Blasius solution with the value of d2fdeta2
# ----------------------------------------------------------------
y0=np.array([0.,0.,derivate[0],1.,derivate[1]])
sol=solve_ivp(ydot,[0,xmax],y0,method='LSODA')

# Plotting of the velocity
# ------------------------

plt.figure()
plt.plot(sol.t,-sol.y[1,:])
plt.xlabel(r'$\tilde{x}$')
plt.ylabel(r'$v/\sqrt{y}$')
plt.xlim(0,10.)

# Plotting of the velocity
# ------------------------

plt.figure()
plt.plot(sol.t,sol.y[3,:])
plt.xlabel(r'$\tilde{x}$')
plt.ylabel(r'$\theta$')
plt.xlim(0,10.)

# Saving of the solution
# ----------------------

N=np.size(sol.t)
A=np.zeros((6,N))
for i in range(N):
    A[0,i]=sol.t[i]
    for k in range(5):
        A[k+1,i]=sol.y[k,i]
    #end for
#end for

np.savetxt('verticalheatingwall.dat',np.transpose(A))


