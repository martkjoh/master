import numpy as np
import sys, os
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
from numpy import sqrt

sys.path.append(sys.path[0] + "/..")

from constants_lattice import m_pi, f_pi
ml, fl = m_pi, f_pi
fl1 = 136/sqrt(2)
fl2 = 130/sqrt(2)
fl = [fl1, fl2]


print(os.getcwd())
f = "data_brandt2/EoS_pi.txt"
muI, nI, E_nI,p,E_p,eps, E_eps, I, E_I = np.loadtxt(f, unpack=True)
muI = muI *2


fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharex=True)
ax[0].set_xlim(.9, 2.1)


ax[0].plot(muI, nI, 'k--')
ax[1].plot(muI, eps, 'k--')
ax[2].plot(muI, p, 'k--')

ax[0].fill_between(muI, nI-E_nI, nI+E_nI, alpha=.6)
ax[1].fill_between(muI, eps-E_eps, eps+E_eps, alpha=.6)
ax[2].fill_between(muI, p-E_p, p+E_p, alpha=.6)




plt.show()