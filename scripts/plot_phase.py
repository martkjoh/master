import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


from constants import m_pi_MeV, m_K_MeV

mK = m_K_MeV / m_pi_MeV
mpi = 1

muS = lambda muI : (-mpi**2 + np.sqrt((mpi**2 - muI**2)**2 + 4*mK**2*muI**2) )/ (2*muI)

fig, ax = plt.subplots(figsize=(6, 6))


muI = np.linspace(0, 4, 500)

# Normal phase
# mu_I

mask1 = muI < mK-mpi/2
ax.plot(mpi*np.ones_like(muI[mask1]), muI[mask1], "k")
ax.plot(mpi*np.ones_like(muI[mask1]), -muI[mask1], "k")
ax.plot(-mpi*np.ones_like(muI[mask1]), muI[mask1], "k")
ax.plot(-mpi*np.ones_like(muI[mask1]), -muI[mask1], "k")

mask2 = muS(muI)> mK-mpi/2
ax.plot(muI[mask2], muS(muI[mask2]), "--k")
ax.plot(muI[mask2], -muS(muI[mask2]), "--k")
ax.plot(-muI[mask2], muS(muI[mask2]), "--k")
ax.plot(-muI[mask2], -muS(muI[mask2]), "--k")

mask3 = mK - muI/2 > mK-mpi/2
ax.plot(muI[mask3], (mK - 1/2*muI)[mask3], "k")
ax.plot(muI[mask3], -(mK - 1/2*muI)[mask3], "k")
ax.plot(-muI[mask3], (mK - 1/2*muI)[mask3], "k")
ax.plot(-muI[mask3], -(mK - 1/2*muI)[mask3], "k")

ax.plot([0, 0], [mK, 10], "--k")
ax.plot([0, 0], [-mK, -10], "--k")


plt.xlim(-2.2, 2.2)
plt.ylim(-5, 5)

plt.text(-0.94, 0, "Normal phase")
plt.text(1, 4, "$\\langle K^+\\rangle$")
plt.text(-1.6, -4.6, "$\\langle K^-\\rangle$")
plt.text(1, -4.6, "$\\langle \\bar K^0\\rangle$")
plt.text(-1.6, 4, "$\\langle K^0\\rangle$")
plt.text(1.2, 0, "$\\langle \pi^+\\rangle$")
plt.text(-1.9, 0, "$\\langle \pi^-\\rangle$")


ax.set_xlabel("$\\mu_I / m_\pi$")
ax.set_ylabel("$\\mu_S / m_\pi$")

fig.savefig("figurer/phase_diagram.pdf", bbox_inches="tight")
