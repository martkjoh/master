import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)

g = np.array([
    0, 
    1,
    7/6,
    3/2,
    5/3,
    np.inf
    ])
n = len(g)
b_func = lambda g: (g - 2) / (3*g - 4)
b = b_func(g)
b[-1] = 1/3 #((inf - 2) / (3 inf - 1))


N = 500

M = np.ones((len(g), N)) * np.linspace(0, 1, N)[None, :]
R = M**b[:, None]
R[R>1e5] = np.nan
norm = 1/np.nanmax(R, axis=1)
R = R *norm[:, None]


fig, ax = plt.subplots(2, figsize=(16, 22))

colors = [cm.plasma(i/(n)) for i in range(n)]

g_label = ["%.2f," % gi if gi!=np.inf else "\\infty,\,\,\,\,\," for gi in g]
b_label = ["%.2f," % bi for bi in b]
labels = ["$\\gamma = "+g_label[i]+" \,\, \\beta ="+b_label[i]+"$" for i in range(n)]

for i, beta in enumerate(b):
    ax[0].plot(R[i], M[i], color=colors[i], label=labels[i], lw=3)

ax[0].legend(loc=2)

gs = np.linspace(0, 3, 200)
bs = b_func(gs)
i = np.argmax(bs) + 1
c="k"
ax[1].plot(gs[:i ], bs[:i], "--", alpha=0.6, color=c)
ax[1].plot(gs[i:], bs[i:], "--", alpha=0.6, color=c, label="$\\beta(\gamma)$")


for i in range(n-1):
    ax[1].plot(g[i], b[i], "x", color=colors[i], ms=10, mew=4, label=labels[i])

ax[1].plot(2.5, 0, "x", ms=10, mew=4, color=colors[-1], label=labels[-1])
ax[1].annotate("", xy=(2.5 + 0.4, 0), xytext=(2.5+0.05, 0), arrowprops=dict(arrowstyle="->"))
ax[1].text(2.5-0.3, -0.2, "$(\gamma=\infty,\,\\beta=\\frac{1}{3})$")
ax[1].set_ylim(-2, 2)
ax[1].legend()


ax[0].set_xlabel("$R$")
ax[0].set_ylabel("$M$")
ax[1].set_xlabel("$\gamma$")
ax[1].set_ylabel("$\\beta$")

fig.savefig("figurer/mass_radius_relation_polytropes.pdf", bbox_inches="tight")

