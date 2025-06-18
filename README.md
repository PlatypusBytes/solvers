![Tests](https://github.com/PlatypusBytes/solvers/actions/workflows/workflow.yml/badge.svg)
[![codecov](https://codecov.io/gh/PlatypusBytes/solvers/graph/badge.svg?token=CRWV3A3WLR)](https://codecov.io/gh/PlatypusBytes/solvers)

# README #

```
     _______.  ______    __      ____    ____  _______ .______       _________.
    /       | /  __  \  |  |     \   \  /   / |   ____||   _  \     /         |
   |   (----`|  |  |  | |  |      \   \/   /  |  |__   |  |_)  |   |   (------`
    \   \    |  |  |  | |  |       \      /   |   __|  |      /     \   \
.----)   |   |  `--'  | |  `----.   \    /    |  |____ |  |\  \------)   |
|_______/     \______/  |_______|    \__/     |_______|| _| `.__________/


Solvers: Numerical solvers for numerically solving partial differential equations
```

Consists of:

* Static solver
* [Newmark solver](https://ascelibrary.org/doi/10.1061/JMCEA3.0000098) (explicit and implicit with consistent mass matrix)
* [Zhai solver](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0207%2819961230%2939%3A24%3C4199%3A%3AAID-NME39%3E3.0.CO%3B2-Y) (explicit with consistent mass matrix)
* Central difference solver (explicit with possibility to lump mass matrix)
* [Noh-Bathe solver](https://www.sciencedirect.com/science/article/abs/pii/S0045794913001934) (explicit with possibility to lump mass matrix)
