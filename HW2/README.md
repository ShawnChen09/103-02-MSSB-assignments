# HW2

This assignment addresses two exercises from the textbook: Exercise 3.7.3 (mathematical derivation) and Exercise 3.7.5 (computational implementation of kinetic models).

## Overview

The assignment consists of two parts:

### Exercise 3.7.3
- Mathematical derivation of the Michaelis-Menten equation from the Mass-action model
- Analysis of quasi-steady-state assumption
- LaTeX document with detailed mathematical steps

### Exercise 3.7.5
- Michaelis-Menten model simulation
- Mass-action model parameter optimization
- Comparison of steady states between the two models
- Visualization of simulation results

## Files

### Exercise 3.7.3
- `3.7.3.tex`: LaTeX source file with mathematical derivations
- `3.7.3.pdf`: Compiled PDF document with the solution

### Exercise 3.7.5
- `main.py`: Contains the main test functions for each part of the assignment
- `models.py`: Implements the Michaelis-Menten and Mass-action kinetic models
- `simulator.py`: Contains the ODE simulator and simulation runner
- `optimization.py`: Implements parameter optimization for the Mass-action model
- `visualization.py`: Contains functions for visualizing simulation results
- `3.7.5.ipynb`: Jupyter notebook demonstrating the implementation and results

## Demo

### Exercise 3.7.3
The mathematical derivation is available in the `3.7.3/3.7.3.pdf` document.

### Exercise 3.7.5

You can run the `HW2-3.7.5.ipynb` Jupyter notebook to see the implementation and results.

## Requirements

### Exercise 3.7.3
- LaTeX compiler (for rebuilding the PDF if needed)

### Exercise 3.7.5
- NumPy
- SciPy
- Matplotlib