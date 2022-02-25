import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve



def LE(x, y, n):
    th0, dth0 = y
    dth = dth0
    ddth = -th0**n - 2/x*dth0
    return np.array([dth, ddth])


fig, ax = plt.subplots(figsize=(20, 10))

for n in range(0, 10):
    s = solve_ivp(LE, (1e-10, 100), (1, 0), args=(n,), max_step=0.1)
    ax.plot(s.t, s.y[0], "k")

ax.set_ylim(-1.1, 1.1)
plt.show()


# n=5
# s = solve_ivp(LE, (1e-10, 8), (1, 0), args=(n,), max_step=0.1, dense_output=True)

# th = lambda x : s.sol(x)[0]

# th(0) 
# x1 = fsolve(th, 1, band=(1, 5))
# print(x1)

