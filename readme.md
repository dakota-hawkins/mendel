# mendel
Genetic Aglorithm framework in Python.

Mendel is a general framework for creating genetic algorithms in Python. While
the general workflow of genetic algorithms can be generalized (i.e. create a
population of set size, represent different parameter values as individuals
within the population, assess performance of individuals given some function
fitness function, "genetically cross" individuals to create new population,
and finally repeat until some stopping criteria), inevitably some
problem-specific routines must be considered.

To account for a generalized framework, while also allowing for
problem-specific implementations, Mendel generalizes the genetic algorithm
process in the `GeneticAlgorithm` and `Individual` classes. Problem specific
methods can then be implemented by extending the `FitnessMixin` class. This
structure allows users to specify their own handling of data input, parameter
unpacking, and performance evaluation, without having to re-implement the more
general aspects of a genetic algorithm.

## Requirements
Mendel requires `Python 3.6`, `numpy`/`scipy` for numerical computations,
`tqdm` for progress bars, and `pytest` for unit testing.

## Tests
To run tests, first install `pytest`, open a terminal, navigate to the
repository head, and issue the command `pytest -v tests/mendel_tests.py`.

## Installation
The repo is not currently available on popular package managers, such as `pip`
and `conda`. To install, please fork this repository. 