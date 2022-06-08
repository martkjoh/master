# master
This repository contains the code, plots, and LaTeX files used for my Master's thesis in physics at Norwegian University of Science and Technology (NTNU) spring 2022.
The thesis is titled "Pion Stars --- Thermodynamic Properties of the Pion condensate using Chiral Perturbation Theory", and is advised by Jens Oluf Andersen.



### Chiral perturbation theory

Chiral perturbation theory is an effective field theory.
It describes quarks and the strong force using mesons, such as pions and kaons, by exploiting the symmetries of QCD to construct an effective Lagrangian,
$$\mathcal L = \frac{1}{4} f^2 {\rm Tr} \\{ \nabla_\mu \Sigma \nabla^\mu \Sigma^\dagger \\} + \frac{1}{4} f^2 {\rm Tr} \\{ \chi^\dagger \Sigma + \Sigma^\dagger \chi \\} ...,$$
where $\Sigma$ is a $SU(N)$-valued field parametrized by the mesons.


### Pion stars

At high isospin-density, QCD transitions from the vacuum phase to a pion-condensed phase.
This condensate is conjectured to be able to form stars, which are modeled by the Tolman-Oppenheimer-Volkoff equation, $$ \frac{{\rm d}p }{{\rm d} r} = - \frac{G m(r) u(r)}{ r^2} \left( 1 + \frac{p(r)}{u(r)}\right) \left( 1 + \frac{4 \pi r^2 p(r)}{m(r)}\right)  \left( 1 - \frac{2 G m(r)}{r^2} \right)^{-1} $$
Preliminary, numerical studies by Brandt et al. supports this conjecture.
In this thesis, we use chiral perturbation theory to investigate the properties of pion stars.


## Layout of repository
There are three main folders in this repository
- `oppgave` contains all LaTeX files used in the thesis, as well as a rendered version of the thesis. (main.pdf)
- `power_expansions` contain the SageMath files for calculating series expansions of the chiral Lagrangian. These are written useing Jupyter notebooks.
- `script` contains the numerical scipts, written in Python.

This repository contains all plots used in the thesis (`scripts/figurer`).
However, the data used to generate these are not included to save space.
The to generate all data, and create the requied and re-render the figures, run `scripts/generate_all_data.py`.
This takes approximately 45 mins on a underpowered laptop.






