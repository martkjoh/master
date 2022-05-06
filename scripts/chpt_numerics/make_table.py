import numpy as np
import sys
from scipy.interpolate import splev, splrep

sys.path.append(sys.path[0] + "/..")
from constants import get_const_pion
from integrate_tov import get_u

u0, m0, r0 = get_const_pion()

def load_data(name=""):
    sols = np.load("pion_star/data/sols"+name+".npy", allow_pickle=True)
    n = 3
    data = [[] for _ in range(n)]
    
    for s in sols:
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

    print("R = ", data[0][j]*r0)
    print(("u_c"+name).ljust(15)+"="+("%.3e"%uc_max).rjust(11))
    print(("p_c"+name).ljust(15)+"="+("%.3e"%pc_max).rjust(11))

def find_mu(name):
    path = "pion_star/data/eos"+name+".npy"
    mu, p, _ = np.load(path)

    data = load_data(name)
    j = np.argmax(data[1])
    pc_max = data[2][j]
    if mu[2]-mu[1]<0:Exception("reverse mu alert!!!")
    tck = splrep(p, mu, s=0, k=1)
    muc_max = splev(pc_max, tck)
    print(("mu_c"+name).ljust(15)+"="+ ("%.3e"%(muc_max-1.)).rjust(11))


names = ["" ,"_nlo","_EM","_e","_mu","_neutrino"]

for name in names:
    find_p_u_c(name)
    find_mu(name)
