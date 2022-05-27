import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt
from matplotlib import cm

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


def p(r, k):
    return - (
        sqrt(1 - 2 * k) - sqrt(1 - 2 * k * r**2 ) 
        ) \
    / (
        3 * sqrt(1 - 2 *  k) - sqrt(1 - 2 * k * r**2)
        )


def rel():
    k0 = 0.42
    k1 = 0.44
    N = 10
    k = np.linspace(k0, k1, N)


    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlabel("$r / r_0$")
    ax.set_ylabel("$p / p_0$")

    ax.set_title("$k_1 \in [" + str( k0 ) + "," + str(k1) + "]$")

    for i in range(N):
        r = np.linspace(0, 1, 100)
        ax.plot(r, p(r, k[i]), color = cm.viridis(i / N), lw=2)

    plt.savefig("figurer/incompressible.pdf", bbox_inches="tight")


def newt():

    k0 = 0.005
    k1 = 0.02

    N = 10
    k = np.linspace(k0, k1, N)


    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlabel("$r / r_0$")
    ax.set_ylabel("$p / p_0$")

    ax.set_title("$k_1 \in [" + str( k0 ) + "," + str(k1) + "]$")

    for i in range(N):
        r = np.linspace(0, 1, 100)
        ax.plot(r, p(r, k[i]), alpha = 0.4, color = cm.viridis(i / (N)), lw=5)
        ax.plot(r, 1/2 * k[i] * (1 - r**2), "--k", lw=2)
    ax.plot(0, 0, "--k", lw=2, label=r"$\frac{1}{2}k_1 (1 - r^2)$")
    plt.legend()
    plt.savefig("figurer/incompressible_newt.pdf", bbox_inches="tight")
    

rel()
newt()
