import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

plt.rc("font", family="serif", size=21)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)

g = np.array([
    0,
    15/12,
    3/2,
    5/3,
    # 3,
    np.inf
    ])
n = len(g)
b_func = lambda g: (g - 2) / (3*g - 4)
b = b_func(g)
b[-1] = 1/3 #((inf - 2) / (3 inf - 1))


N = 500

M = np.ones((len(g), N)) * np.linspace(0, 1, N)[None, :]
R = M**b[:, None]
R[R>1e2] = np.nan
norm = 1/np.nanmax(R, axis=1)
R = R *norm[:, None]


fig, ax = plt.subplots(2, 1, figsize=(10, 13))

colors = [cm.plasma(i/(n)) for i in range(n)]

g_label = ["%.2f," % gi if gi!=np.inf else "\\infty,\,\,\,\,\," for gi in g]
b_label = ["%.2f," % bi for bi in b]
labels = ["$\\gamma = "+g_label[i]+" \,\, \\beta ="+b_label[i]+"$" for i in range(n)]

for i, beta in enumerate(b):
    ax[1].plot(R[i], M[i], color=colors[i], label=labels[i], lw=2)

# ax[0].legend(loc=2)

gs = np.linspace(0, 3, 200)
bs = b_func(gs)
i = np.argmax(bs) + 1
c="k"
ax[0].plot(gs[:i ], bs[:i], "--", alpha=0.6, color=c)
ax[0].plot(gs[i:], bs[i:], "--", alpha=0.6, color=c, label="$\\beta(\gamma)$")


for i in range(n-1):
    ax[0].plot(g[i], b[i], "x", color=colors[i], ms=10, mew=4, label=labels[i])

ax[0].plot(2.5, 0.3, "x", ms=10, mew=4, color=colors[-1], label=labels[-1])
ax[0].annotate("", xy=(2.5 + 0.4, 0.3), xytext=(2.5+0.05, 0.3), arrowprops=dict(arrowstyle="->"))
# ax[0].text(2.5-0.3, -0.3, "$(\gamma=\infty,\,\\beta=\\frac{1}{3})$")
ax[0].set_ylim(-1.4, 3.9)
ax[0].set_xlim(-0.04, 3)


ax[0].legend()


ax[1].set_xlabel("$R$")
ax[1].set_ylabel("$M$")
ax[0].set_xlabel("$\gamma$")
ax[0].set_ylabel("$\\beta$")

# plt.show()
fig.savefig("figurer/mass_radius_relation_polytropes.pdf", bbox_inches="tight")

