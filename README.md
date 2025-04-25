# GenDynDiff: Velocity Diffusion with GemNetT

**GenDynDiff** is a deep generative diffusion framework for atomistic systems, built on PyTorch Lightning and GemNetT.  
This fork is tailored for **velocity field generation and denoising** (not full structure generation), using real atomic numbers and lattice parameters from your MD simulation data.

---

## Features

- **Velocity Diffusion:** Trains a model to denoise and generate atomic velocities from MD trajectories.
- **Supports Real Lattice and Atomic Numbers:** Ensures correct periodic boundary handling and atomic embeddings.
- **PyTorch Lightning Integration:** Robust training, logging, and checkpointing.
- **Trajectory Sampling:** Generate new velocity fields from a trained checkpoint.

---

## Environment Setup

This repository is tested on **Ubuntu** using **Poetry** and **Python 3.12.3**.

### 1. Clone the Repository


### 2. Install Poetry (if not already installed)


### 3. Set Up the Poetry Environment


**Note:**  
- All dependencies (including PyTorch, PyTorch Lightning, ASE, pymatgen, hydra, tqdm, and wandb) are managed by Poetry.

### 4. Activate the Environment

### 5. uv pip install -e . to install all dependencies if you haven't

#### When you run the program, make sure to change the dump.yaml in conf/data to your device's parameters

#### to run, run gendyndiff/run.py 
