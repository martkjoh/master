import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton


from matplotlib import cm

plt.rc("font", family="serif", size=20)
plt.rc("mathtext", fontset="cm")
plt.rc("lines", lw=2)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=1)


eq = lambda x, y: A * (y**2-1)**(3/2) - x*(1-1/x**4)

N = 100
x = np.linspace(1, 5, 100)
y = newton(lambda y : eq(x, y), x0=2*np.ones(N))

plt.plot(x, y)
plt.show()
