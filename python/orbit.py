import numpy as np


def planetorbit(Mstarsun,periodday,e,n=10000):
    """
    PROGRAM Orbit

    General Description:
    ====================
    Orbit computes the orbit of a small mass about a much larger mass, 
    or it can be considered as computing the motion of the reduced mass 
    about the center of mass.
 
    "An Introduction to Modern Astrophysics", Appendix J
    Bradley W. Carroll and Dale A. Ostlie
    Second Edition, Addison Wesley, 2007
 
    Weber State University
    Ogden, UT
    modastro@weber.edu

    Translated to Python by D. Nidever  March 2023

    Parameters
    ----------
    Mstarsun : float
       Mass of the parent star (in solar masses).
    periodday : float
       Period in days.
    e : float
       Eccentricity.
    n : int, optional
       Number of time steps.  Default is 10000.

    Returns
    -------
    tab : table
       Table of computed orbit values.

    Example
    -------

    tab = planetorbit(1.0,1.0,0.9)

    """

    G = 6.673e-11         # m^3 / kg /s^2
    AU = 1.4959787066e11  # m
    M_Sun = 1.9891e30     # kg
    yr = 3.15581450e7     # sec
    two_pi = 2*np.pi
    eps_dp = 1e-15

    # Convert entered values to conventional SI units
    Mstar = Mstarsun*M_Sun
    P = periodday * 3600 * 24.

    # Calculate the orbital period in seconds using Kepler's Third Law (Eq. 2.37)
    # P = np.sqrt(4*np.pi*np.pi*a*a*a/(G*Mstar))
    # calculate semi-major axis from the period
    a = ((P**2*G*Mstar)/(4*np.pi*np.pi))**(1/3)
    aAU = a/AU
    
    n += 1    # increment to include t=0 (initial) point

    # Initialize print counter, angle, elapsed time, and time step.
    k = 1              # printer counter
    theta = 0.0        # angle from direction to perihelion (radians)
    t = 0.0            # elapsed time (s)
    dt = P/(n-1)       # time step (s)
    delta = 2*eps_dp   # allowable error at end of period
    # r, LoM, dtheta;  # declare variables used inside loop
    r = 0.0
    LoM = 0.0
    dtheta = 0.0
    
    # Initialize output table
    dtype = [('niter',int),('t',float),('tyr',float),('r',float),('rAU',float),
             ('theta',float),('x',float),('y',float),
             ('LoM',float),('drdt',float),('dxdt',float),('dydt',float)]
    tab = np.zeros(n,dtype=np.dtype(dtype))
    
    # Start main time step loop
    count = 0
    while (((theta - two_pi) < dtheta/2) and ((t - P) < dt/2)):

        # Calculate the distance from the principal focus using Eq. (2.3); Kepler's First Law.
        r = a*(1 - e*e)/(1 + e*np.cos(theta))

        tab['niter'][count] = count+1
        tab['t'][count] = t
        tab['tyr'][count] = t/yr
        tab['r'][count] = r
        tab['rAU'][count] = r/AU        
        tab['theta'][count] = np.rad2deg(theta)
        tab['x'][count] = r*np.cos(theta)
        tab['y'][count] = r*np.sin(theta)
        
        # Prepare for the next time step:  Update the elapsed time.
        t += dt

        # Calculate the angular momentum per unit mass, L/m (Eq. 2.30).
        LoM = np.sqrt(G*Mstar*a*(1 - e*e))
        tab['LoM'][count] = LoM

        # Compute the next value for theta using the fixed time step by combining
        #     Eq. (2.31) with Eq. (2.32), which is Kepler's Second Law.
        dtheta = LoM/(r*r)*dt
        theta += dtheta

        count += 1

    # Calculate derivatives
    tab['drdt'] = np.gradient(tab['r'])/dt
    tab['dxdt'] = np.gradient(tab['x'])/dt
    tab['dydt'] = np.gradient(tab['y'])/dt    
    
    return tab


def starorbit(Mstarsun,Mplanetearth,periodday,e,n=10000):
    """
    Compute orbit of star due to small planet using
    ORBIT code from Carroll & Ostlie

    Parameters
    ----------
    Mstarsun : float
       Mass of the parent star (in solar masses).
    Mplanetearth : float
       Planet mass (in earth masses).
    periodday : float
       Period in days.
    e : float
       Eccentricity.
    n : int, optional
       Number of time steps.  Default is 10000.

    Returns
    -------
    tab : table
       Table of computed star orbit values.

    Example
    -------

    tab = starorbit(1.0,1.0,365,0.1)

    """

    G = 6.673e-11           # m^3 / kg /s^2
    AU = 1.4959787066e11    # m
    M_Sun = 1.9891e30       # kg
    M_Jupiter = 1.89813e27  # kg
    M_Earth  = 5.9736e24    # kg
    yr = 3.15581450e7       # sec
    two_pi = 2*np.pi
    eps_dp = 1e-15
    
    # Compute the orbit of the star

    # Convert entered values to conventional SI units
    Mstar = Mstarsun * M_Sun
    Mplanet = Mplanetearth * M_Earth
    period = periodday * 3600*24
    
    # The reduced mass is
    #  mu = m1*m2/(m1+m2)
    # if m1 >> m2, then
    #  mu = m2 (smaller mass)

    # Calculate the orbital period in seconds using Kepler's Third Law (Eq. 2.37)
    # P = 4*pi^2*a^3/(G*(m1+m2))    
    # P = np.sqrt(4*np.pi*np.pi*a*a*a/(G*Mstar))
    # calculate semi-major axis from the period
    a = (period**2*G*(Mstar+Mplanet)/(4*np.pi*np.pi))**(1/3)
    aAU = a/AU
    
    # Get the planet's (reduced mass) orbit
    ptab = planetorbit(Mstarsun,periodday,e,n=n)

    # Convert planet values to star values
    # m1/m2 = a2/a1 = v2/v1
    # r1 = m2/m1 * r2
    # Initialize output table
    dt = ptab['t'][1]-ptab['t'][0]
    dtype = [('niter',int),('t',float),('tyr',float),('r',float),('rAU',float),
             ('theta',float),('x',float),('LoM',float),('drdt',float),('rv',float)]
    stab = np.zeros(len(ptab),dtype=np.dtype(dtype))
    stab['niter'] = ptab['niter']
    stab['t'] = ptab['t']
    stab['tyr'] = ptab['tyr']        
    stab['r'] = (Mplanet/Mstar)*ptab['r']
    stab['rAU'] = (Mplanet/Mstar)*ptab['rAU']
    stab['theta'] = 180 + ptab['theta']
    stab['x'] = -(Mplanet/Mstar)*ptab['x']
    stab['drdt'] = (Mplanet/Mstar)*ptab['drdt']
    stab['rv'] = np.gradient(stab['x'])/dt

    return stab
