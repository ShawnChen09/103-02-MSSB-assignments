# HW1

This assignment implements various numerical methods for solving the ordinary differential equation (ODE) dx/dt = 1 - x.

## Overview

The implementation includes:
- Analytical solution
- Python's built-in ODE solver (scipy.integrate.odeint)
- Forward Euler method
- 4th-order Runge-Kutta method (RK4)

## Files

- `analytical.py`: Contains the analytical solution and model definition
- `numerical_methods.py`: Implements Euler and RK4 methods
- `solvers.py`: Contains the general solver function
- `test_methods.py`: Contains test functions for different numerical methods
- `main.ipynb`: Jupyter notebook demonstrating the implementation and results

## Demo

The implementation and results are demonstrated in the `main.ipynb` Jupyter notebook. Run the notebook cells in order to see the comparison of different numerical methods with the analytical solution.

## Requirements

- NumPy
- SciPy
- Matplotlib

## Author

Student Name: 陳善恩 Chen Shan En
Student ID: R13621202