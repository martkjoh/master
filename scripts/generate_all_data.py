import os, sys

path = sys.path[0]


def run(name, kwd = ""):
    # If you do not have the alias python3, change to path to python 3 file
    cmd = "python3 " + path + "/" + name + ".py"
    os.system(cmd + " " + kwd)


def set_lattice(state="False"):
    """
    Rewrites the "constants.py" file to enable lattice constants.
    I know it is hacky, it got added on much later...
    """
    f = open("constants.py", "r")
    txt = f.read()
    txt = txt.split("\n")
    txt[4] = "lattice = " + state
    txt = "\n".join(txt)
    f = open("constants.py", "w")
    f.write(txt)
    f.close()


set_lattice()

run("plot_polytrope")
run("incompressible_fluid/incompressible_fluid")
run("chpt_numerics/plot_surface")
run("chpt_numerics/plot_masses")
run("chpt_numerics/plot_phase_diagram")
run("fermi_gas_star/fermi_gas_eos")
run("fermi_gas_star/fermi_gas")
run("fermi_gas_star/plot")

# Generate pion condensate eos
run("chpt_numerics/chpt_eos")
run("chpt_numerics/chpt_lepton_eos")
run("chpt_numerics/neutrino_eos")
run("chpt_numerics/free_energy_nlo")
run("chpt_numerics/neutrino_eos_nlo")


set_lattice("True")

run("chpt_numerics/neutrino_eos")
run("chpt_numerics/free_energy_nlo")
run("chpt_numerics/neutrino_eos_nlo")


set_lattice()


# integrate pion stars
# This is going to take some time...
# Comment out parts of the script if 
# you know what you want
run("pion_star/pion_star")

# plots
run("pion_star/plot")
run("chpt_numerics/brandt_plot_eos2")
run("chpt_numerics/brandt_plot_RM")


