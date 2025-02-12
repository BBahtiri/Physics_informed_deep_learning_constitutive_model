# Physics-Informed Deep Learning for Thermodynamically Consistent Material Modeling

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

A neural architecture that learns constitutive material behavior while rigorously enforcing thermodynamic consistency through built-in physical constraints.

## Key Features

🔬 **Physics-Informed Architecture**  
- Enforces fundamental thermodynamic principles:  
  - Non-negative free energy (ψ ≥ 0)  
  - Stress potential relationship (σ = ∂ψ/∂ε)  
  - Non-negative dissipation (D ≥ 0)  
  - Frame indifference through invariant formulation  
  - Proper reference configuration (zero initial conditions)  

🧠 **Network Components**  
- LSTM layers for history-dependent material behavior  
- PICNN (Partially Input Convex Neural Network) for convex free energy  
- Internal state variables with thermodynamically consistent evolution  
- Automatic stress computation via chain rule differentiation  
- Custom loss functions combining data fidelity and physical constraints  

📊 **Data Integration**  
- Processes experimental stress-strain data from Excel files  
- Automatic calculation of 6 strain invariants (I₁-I₆)  
- Derivative computation for stress relationships  
- Comprehensive data normalization/visualization pipeline  
- Batch training with adaptive learning rate scheduling  

## Installation
bash
git clone https://github.com/yourusername/PIDL-Material-Modeling.git
cd PIDL-Material-Modeling
pip install -r requirements.txt

## Usage

1. **Data Preparation**  
   Format experimental data in Excel with columns:  
   - Strain components: `E11, E12, E13, E22, E23, E33`  
   - Stress components: `S11, S12, S13, S22, S23, S33`  
   - Timestep: `DT`  


## Results

The model generates:  
- Stress-strain predictions vs ground truth comparisons  
- Thermodynamic consistency validation plots  
- Training loss curves (stress error, dissipation, energy)  
- Model checkpoints for deployment  
- Detailed visualizations of all network outputs  

