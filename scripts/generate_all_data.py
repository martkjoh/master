import os, sys

path = sys.path[0]


def run(name, kwd = ""):
    # If you do not have the alias python3, change to path to python 3 file
    cmd = "python3 " + path + "/" + name + ".py"
    os.system(cmd + " " + kwd)


def set_lattice(state="False"):
    f = open("constants.py", "r")
    txt = f.read()
    txt = txt.split("\n")
    txt[4] = "lattice = " + state
    txt = "\n".join(txt)
    f = open("constants.py", "w")
    f.write(txt)
    f.close()


set_lattice()

run("incompressible_fluid/incompressible_fluid") 
run("fermi_gas_star/fermi_gas_eos")
run("fermi_gas_star/fermi_gas")

# # Generate pion condensate eos
run("chpt_numerics/chpt_eos")
run("chpt_numerics/chpt_lepton_eos")
run("chpt_numerics/neutrino_eos")
run("chpt_numerics/free_energy_nlo")
run("chpt_numerics/neutrino_eos_nlo")
run("chpt_numerics/free_energy_nlo")
run("chpt_numerics/neutrino_eos_nlo")
run("chpt_numerics/brandt_plot_eos2")
run("chpt_numerics/brandt_plot_RM")

set_lattice("True")

run("chpt_numerics/chpt_eos")
run("chpt_numerics/chpt_lepton_eos")
run("chpt_numerics/neutrino_eos")
run("chpt_numerics/free_energy_nlo")
run("chpt_numerics/neutrino_eos_nlo")
run("chpt_numerics/free_energy_nlo")
run("chpt_numerics/neutrino_eos_nlo")
run("chpt_numerics/brandt_plot_eos2")
run("chpt_numerics/brandt_plot_RM")

set_lattice()


# integrate pion stars
run("pion_star/pion_star")

# plots
run("pion_star/plot")
run("chpt_numerics/plot_surface")
run("chpt_numerics/plot_masses")
run("chpt_numerics/plot_phase_diagram")

