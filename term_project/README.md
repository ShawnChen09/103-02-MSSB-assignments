# Environmental DNA Modeling in Mesopelagic Ocean

This project reproduces and extends the research presented in the paper "Allan, E. A., DiBenedetto, M. H., Lavery, A. C., Govindarajan, A. F., & Zhang, W. G. (2021). Modeling characterization of the vertical and temporal variability of environmental DNA in the mesopelagic ocean. Scientific Reports, 11(1), 21273".

## Project Overview

### 1. Reproduction of Original Research
This project attempts to reproduce partial results from the original article by implementing the core mathematical models described in the paper.

### 2. Parameter Space Exploration
Examining how key parameters affect eDNA distribution patterns in the water column, focusing on:
- Vertical distribution
- Temporal dynamics
- Environmental factors affecting degradation

### 3. Model Extension: eDNA Transportation via Predator Fecal Matter
Adding a new reaction pathway to the original model to consider eDNA transportation through predator fecal matter and its potential effects on distribution patterns.

### 4. Migration Pattern Estimation
Investigating whether migration patterns and species composition can be estimated from mixed eDNA data through signal decomposition techniques.

## Repository Structure

- `data/`: Contains datasets for model parameters
- `edna_model/`: Core implementation of the eDNA models
- `examples/`: Example scripts demonstrating model usage

## Getting Started

To run the simulations, see the example scripts in the `examples/` directory.