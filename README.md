# master
This repository contains the code, plots, and LaTeX files used for my Master's thesis in physics at Norwegian University of Science and Technology (NTNU) spring 2022.
The thesis is titled "Pion Stars --- Thermodynamic Properties of the Pion condensate using Chiral Perturbation Theory", and my advisor is Jens Oluf Andersen.

## Chiral perturbation theory

Chiral perturbation theory is an effective field theory.
It describes quarks and the strong force using mesons, such as pions and kaons, using and effective Lagrangian
$
  \mathcal L = \frac{1}{4} f \mathrm{TR} \{\nabla_\mu \Sigma \nabla^\mu \Sigma^\dagger \} + ...,
$
where $\Sigma$ is a $SU(N)$-valued field parametrized by the mesons.

## Pion stars

At high isospin-density, QCD transitions from the vacuum phase to a pion-condensed phase.
This condensate is conjectured to be able to form stars.
Preliminary, numerical studies by Brandt et al. supports this conjecture.
In this thesis, we use chiral perturbation theory to investigate the properties of pion stars.


### Layout of repository
There are three main folders in this repository
- `oppgave` contains all LaTeX files used in the thesis, as well as a rendered version of the thesis. (main.pdf)
- `power_expansions` contain the SageMath files for calculating series expansions of the chiral Lagrangian. These are written useing Jupyter notebooks.
- `script` contains the numerical scipts, written in Python.

This repository contains all plots used in the thesis (`scripts/figurer`).
However, the data used to generate these are not included to save space.
The to generate all data, and create the requied and re-render the figures, run `scripts/generate_all_data.py`.
This takes approximately (???) mins on a underpowered laptop.



