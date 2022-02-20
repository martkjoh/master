import numpy as np
import sys

sys.path.append(sys.path[0] + "/..")
from integrate_tov import get_u, integrate

u = get_u("pion_star/data/eos.npy")
p = 10**np.linspace(-6, 6, 10)

# print([u(p0) for p0 in p])

uEM = get_u("pion_star/data/eos_EM.npy")
p = 10**np.linspace(-6, 6, 10)

# print([u(p0) for p0 in p])

p = 1e-5
print(u(p))
print(uEM(p))