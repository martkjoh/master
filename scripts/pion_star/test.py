import numpy as np
import sys
import matplotlib.pyplot as plt
from time import time

from scipy.integrate import solve_ivp

sys.path.append(sys.path[0] + "/..")
from integrate_tov import get_u, integrate, dmdr_general, dpdr_general

dmdr = lambda r, y, args: dmdr_general(u, r, y, args) 
dpdr = lambda r, y, args: dpdr_general(u, r, y, args)
f = lambda r, y, args: (dpdr(r, y, args), dmdr(r, y, args))

def stop(r, y, args):
    """Termiation cirerion, p(r) = 0"""
    p, m = y
    return p
stop.terminal = True # attribute for solve_ivp

def stop_time(r, y, args):
    p, m = y
    print("p = ", p )
    return 1

u = get_u("pion_star/data/eos.npy")


max_step=1e-3
pc = 1e-6
r_max = 1e2

t = time()
sol = solve_ivp(
    f, 
    (0, r_max), 
    (pc, 0), 
    args=((pc,),), 
    events=(stop, stop_time),
    max_step=max_step,
)


p, m = sol.y
r = sol.t

# print(sol)
print(time()-t)

plt.plot(r, p)

plt.show()
