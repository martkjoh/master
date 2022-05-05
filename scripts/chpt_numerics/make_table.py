import numpy as np
import sys

sys.path.append(sys.path[0] + "/..")
from constants import get_const_pion
from integrate_tov import get_u

u0, m0, r0 = get_const_pion()


def load_data(name=""):
    sols = np.load("pion_star/data/sols"+name+".npy", allow_pickle=True)
    n = 3
    data = [[] for _ in range(n)]

    for i, s in enumerate(sols):
        data[0].append(s["R"])
        data[1].append(s["M"])
        data[2].append(s["pc"])

    return data


def find_p_u_c(name):
    u_path = "pion_star/data/eos"+name+".npy"
    u = get_u(u_path)

    data = load_data(name)
    j = np.argmax(data[1])
    pc_max = data[2][j]
    uc_max = u(pc_max)

    print(pc_max)
    print(uc_max)


# find_p_u_c("")
# find_p_u_c("_e")
find_p_u_c("_mu")
