# Physics-Informed Deep Learning for Thermodynamically Consistent Material Modeling

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

A neural architecture that learns constitutive material behavior while rigorously enforcing thermodynamic consistency through built-in physical constraints.

## Key Features

🔬 **Material-Specific Formulation**
- **Transversely Isropic Behavior**:
  - Structural tensor formulation (A = a₀ ⊗ a₀)
  - Specialized invariant set for TI materials:
    - I₁ = tr(ε)
    - I₂ = tr(ε²)
    - I₃ = tr(Aε)
    - I₄ = tr(Aε²)
    - I₅ = tr(ε³)
    - I₆ = -2√det(2ε + I)
  - Frame-indifferent stress response through invariant formulation

🌐 **Invariant-Based Architecture**
- Processes strain invariants instead of full tensor
- Avoids tensor operations through scalar invariant formulation
- Automatic derivative calculations for:
  - ∂I₁/∂ε = I
  - ∂I₂/∂ε = 2ε
  - ∂I₃/∂ε = A
  - ∂I₄/∂ε = Aε + εA
  - ∂I₅/∂ε = 3ε²
  - ∂I₆/∂ε = -J·C⁻¹

🧠 **Network Components**  
- LSTM layers for history-dependent behavior in TI materials  
- PICNN architecture for convex free energy in invariant space  
- Internal variables capturing anisotropic hardening  
- Stress computation via invariant chain rule:  
  ```math
  S = 2∑_{i=1}^6 (∂ψ/∂I_i)(∂I_i/∂ε)
  ```

📊 **Data Integration**  
- Processes experimental stress-strain data from Excel files  
- Automatic calculation of 6 strain invariants (I₁-I₆)  
- Derivative computation for stress relationships  
- Comprehensive data normalization/visualization pipeline  
- Batch training with adaptive learning rate scheduling  

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

# References

For more informations, refer to our paper:

[Paper](https://doi.org/10.1016/j.cma.2024.117038)

```
@article{bahtiri2024thermodynamically,
  title={A thermodynamically consistent physics-informed deep learning material model for short fiber/polymer nanocomposites},
  author={Bahtiri, Betim and Arash, Behrouz and Scheffler, Sven and Jux, Maximilian and Rolfes, Raimund},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={427},
  pages={117038},
  year={2024},
  publisher={Elsevier}
}
```

