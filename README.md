# `dmet_excitation_preview`

The demo of code that of the project "Towards Excitations and Dynamical Quantities in Correlated Lattices with Density Matrix Embedding Theory".

We show the fundamental framework of the implementation, and use the exact diagnonalization (or full configuration interaction (FCI) in quantum chemistry language) for impurity solver to demonstrate more directly about the principle.

- The preparation of the impurity problems for chain lattice (`chain_lattice.py`)
- Ground-state DMET, generation of excitation operators, local excitation bases, effective Hamiltonian and the calculation of the excitation energies (`main.py`)

Please see `./example/notebook.ipynb` for the usage for the example of 1D Hubbard model.

## Prerequisites
`pyscf >= 2.0.0`.

## Reference
Shuoxue Li, Chenghan Li, Huanchen Zhai and Garnet Kin-Lic Chan, Towards Excitations and Dynamical Quantities in Correlated Lattices with Density Matrix Embedding Theory. https://arxiv.org/abs/2503.08880 .
