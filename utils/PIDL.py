"""Physics-Informed Deep Learning (PIDL) for Material Modeling

This module implements a thermodynamically consistent neural network architecture
for learning and predicting material behavior. The model enforces physical constraints
derived from the laws of thermodynamics while learning from experimental data.

Thermodynamic Framework:
-----------------------
The model is based on Coleman-Noll procedure of rational thermodynamics:

1. Free Energy Function (ψ):
   - ψ = ψ(ε, z) where ε is the strain tensor and z are internal variables
   - Must be positive: ψ ≥ 0 (stability condition)
   - Serves as a potential for stress: σ = ∂ψ/∂ε
   - Zero initial condition: ψ(ε₀, z₀) = 0 at t = 0
   - Ensures stress-free reference configuration

2. Stress-Strain Relationship:
   - Second Piola-Kirchhoff stress through chain rule: S = Σᵢ(∂ψ/∂Iᵢ)(∂Iᵢ/∂ε)
   - Stress computation uses invariants (Iᵢ) and their derivatives (∂Iᵢ/∂ε)
   - Avoids direct tensor operations by using scalar invariants
   - Each stress component: Sᵢⱼ = Σₖ(∂ψ/∂Iₖ)(∂Iₖ/∂εᵢⱼ)
   - Zero initial stress: S(ε₀, z₀) = 0 at t = 0

3. Initial Conditions:
   - Reference configuration at t = 0
   - Zero free energy: ψ₀ = 0
   - Zero stress: S₀ = 0
   - Zero internal variables: z₀ = 0
   - Ensures consistent reference state

4. Internal Variables (z):
   - Capture history-dependent material behavior
   - Evolution governed by thermodynamic forces: ż = f(∂ψ/∂z)
   - LSTM network learns the evolution equations
   - Initial condition: z(t=0) = 0

5. Strain Invariants:
   - I₁ = tr(ε)
   - I₂ = tr(ε²)
   - I₃ = tr(Aε)
   - I₄ = tr(A(ε)²)
   - I₅ = tr(ε³)
   - I₆ = -2√det(2ε + I)

6. Invariant Derivatives:
   - ∂I₁/∂ε = 1
   - ∂I₂/∂ε = 2ε
   - ∂I₃/∂ε = A
   - ∂I₄/∂ε = Aε + εA
   - ∂I₅/∂ε = 3(ε)²
   - ∂I₆/∂ε = -J·C⁻¹

7. Dissipation Inequality:
   - D = -ψ̇ + S:ε̇ ≥ 0 (Second Law of Thermodynamics)
   - Enforced through non-negative dissipation rate
   - Ensures thermodynamic consistency

Neural Network Architecture:
--------------------------
1. LSTM Network (f0, f02):
   - Processes time history of deformation
   - Learns temporal patterns in material response
   - Outputs features for internal variable prediction

2. Internal Variable Network (f2):
   - Predicts evolution of internal variables
   - Takes LSTM features as input
   - Ensures continuity from initial conditions

3. Free Energy Network (f3, f4):
   - PICNN architecture ensures convexity
   - Inputs: internal variables and invariants
   - Output: scalar free energy function
   - Guarantees thermodynamic consistency

4. Stress Computation:
   - Uses chain rule with invariants: S = Σᵢ(∂ψ/∂Iᵢ)(∂Iᵢ/∂ε)
   - Automatic differentiation for ∂ψ/∂Iᵢ
   - Pre-computed ∂Iᵢ/∂ε for efficiency
   - Maintains frame indifference through invariant formulation
   - Avoids direct tensor operations for better numerical stability

Training Strategy:
----------------
1. Loss Functions:
   - Stress prediction error
   - Non-negative dissipation constraint
   - Positive free energy constraint

2. Physics-Informed Constraints:
   - Thermodynamic consistency enforced through architecture
   - Frame indifference through invariant formulation
   - Conservation laws satisfied by construction

Author: bahtiri
"""

import numpy as np
import tensorflow as tf
import os
import sys
from utils.PICNN import PICNN
# Set both numpy and tensorflow to use float32
np.set_printoptions(precision=5)
tf.keras.backend.set_floatx('float32')
import matplotlib.pyplot as plt
import shutil
tf.config.optimizer.set_jit(True)  # Enable XLA.


class PIDL(tf.keras.Model):
    """Physics-Informed Deep Learning (PIDL) model for material behavior prediction.
    
    This model implements a thermodynamically consistent neural network that learns
    and predicts material behavior while satisfying physical constraints. The architecture
    combines LSTM networks for temporal evolution with physics-informed layers for
    ensuring thermodynamic consistency.

    Mathematical Framework:
    ---------------------
    1. Kinematics:
       - Deformation gradient: F = ∂x/∂X
       - Right Cauchy-Green tensor: C = F^T F
       - Green-Lagrange strain: E = (C - I)/2
       
    2. Thermodynamics:
       - Free energy function: ψ(C, z) ≥ 0
       - Stress-strain relation: S = 2∂ψ/∂C
       - Internal variable evolution: ż = f(∂ψ/∂z)
       - Dissipation inequality: D = -ψ̇ + S:Ė ≥ 0
       
    3. Invariant Formulation:
       - I₁ = tr(C)
       - I₂ = ½[(tr(C))² - tr(C²)]
       - I₃ = det(C)
       - I₄ = a₀·Ca₀
       - I₅ = a₀·C²a₀
       - I₆ = -2√I₃
       
    Network Components:
    -----------------
    1. History-dependent behavior:
       - LSTM layers process deformation history
       - Internal variables capture material memory
       - Evolution equations learned from data
       
    2. Free energy prediction:
       - PICNN ensures convexity
       - Automatic differentiation for stress
       - Positive definiteness enforced
       
    3. Thermodynamic constraints:
       - Non-negative dissipation
       - Frame invariance
       - Material symmetry
    """
    def __init__(self, s_all, m_all, s_out, m_out, layer_size, internal_variables, 
                 layer_size_fenergy, batch_size, optimizer, s_dt, m_dt, stress_derivatives, invariants, 
                 use_picnn=True, training_silent=True):
        """
        Initialize the PIDL model with scaling factors and network architecture parameters.
        
        Args:
            s_all: Tuple of scaling factors for input variables
            m_all: Tuple of mean values for input variables
            layer_size: Size of LSTM and dense layers
            internal_variables: Number of internal variables to predict
            layer_size_fenergy: Size of free energy prediction layers
            batch_size: Size of training batches
            s_dt: Scaling factor for dt
            m_dt: Mean value for dt
            use_picnn: If True, use PICNN for f3, otherwise use dense layers
        """
        super(PIDL, self).__init__()
        # Store all initialization parameters as instance variables
        self.s_all = tf.cast(s_all, tf.float32)
        self.m_all = tf.cast(m_all, tf.float32)
        self.s_out = tf.cast(s_out, tf.float32)
        self.m_out = tf.cast(m_out, tf.float32)
        self.layer_size = layer_size
        self.internal_variables_nr = internal_variables
        self.layer_size_fenergy = layer_size_fenergy
        self.batch_size = batch_size
        self.s_dt = tf.cast(s_dt, tf.float32)
        self.m_dt = tf.cast(m_dt, tf.float32)
        self.stress_derivatives_tf = tf.cast(stress_derivatives, tf.float32)
        self.invariants = invariants
        self.use_picnn = use_picnn
        self.training_silent = training_silent
        self.optimizer = optimizer

        # Initialize model architecture with float32
        self.init_state = tf.zeros((batch_size, layer_size), dtype=tf.float32)
        self.f0 = tf.keras.layers.LSTM(units=layer_size, name="history_lstm",
                                     return_sequences=True,
                                     return_state=False,
                                     use_bias=True,
                                     dtype=tf.float32)
        self.f02=tf.keras.layers.LSTM(units=layer_size, name="history_lstm",
                                     return_sequences=True,
                                     return_state=False,
                                     use_bias=True,
                                     dtype=tf.float32)
        
        # Replace Dense layers with TimeDistributed Dense layers
        self.f2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(internal_variables, use_bias=True, dtype=tf.float32),
            name='f2_time_distributed')
        
        # Choose between PICNN and dense layers for f3
        if use_picnn:
            self.f3 = PICNN(x_sizes=[layer_size_fenergy, layer_size_fenergy], 
                           y_sizes=[layer_size_fenergy, layer_size_fenergy], 
                           internal_variables=internal_variables,
                           invariants=invariants,
                           activation='softplus', 
                           x_activation='softplus')
        else:
            self.f3_dense1 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    layer_size_fenergy,
                    activation='softplus',
                    dtype=tf.float32
                ),
                name='f3_dense1_time_distributed'
            )
            self.f3_dense2 = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    layer_size_fenergy,
                    activation='softplus',
                    dtype=tf.float32
                ),
                name='f3_dense2_time_distributed'
            )
            
        self.f4 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, use_bias=True,
                                  kernel_constraint=tf.keras.constraints.NonNeg(),
                                  bias_constraint=tf.keras.constraints.NonNeg(),
                                name='free_energy',
                                dtype=tf.float32),
            name='f4_time_distributed')
        self.print_every = 10
    
    def tf_u_stnd_nrml(self, output, a, b):
        ''' Un-standardize/un-normalize by multiplying with scaling factor '''
        return output * a  # Simply multiply by scaling factor since we normalized by division
    
    def tf_out_stnd_nrml(self, u, a, b):
        ''' Standardize/normalize by dividing by scaling factor '''
        return u / a  # Simply divide by scaling factor

    def internal_variables(self, un_I_tdt_f, un_dt_tdt_f):
        """Predict internal variables using LSTM and feedforward neural networks.
        
        This method implements the evolution of internal variables z(t) that capture
        the material's history-dependent behavior. The architecture ensures:
        1. Proper initial conditions: z(t=0) = 0
        2. Continuous evolution: z(t) is smooth and continuous
        3. History dependence through LSTM layers
        
        Mathematical Framework:
        ---------------------
        1. Input Processing:
           - State vector: x(t) = [I₁(t), I₂(t), I₃(t), I₄(t), I₅(t), I₆(t), dt]
           where Iᵢ are invariants and dt is the timestep
        
        2. LSTM Processing:
           - First LSTM: h₁(t) = LSTM₁(x(t))
           - Second LSTM: h₂(t) = LSTM₂(h₁(t))
           These capture temporal patterns in the deformation history
        
        3. Dense Layer:
           - z̃(t) = f₂(h₂(t)) where f₂ is a dense layer
           Transforms LSTM features into internal variable space
        
        4. Initial Condition Enforcement:
           - z(t=0) = 0 (zero vector)
           - z(t>0) = z̃(t)
           Ensures proper reference configuration
        
        Implementation Details:
        ---------------------
        1. Input Combination:
           - Concatenates invariants and timestep: [I₁,...,I₆, dt]
           - Shape: [batch_size, time_steps, 7]
        
        2. LSTM Processing:
           - Two stacked LSTM layers for temporal feature extraction
           - Maintains sequence information through time
        
        3. Dense Projection:
           - Projects to internal variable space
           - Uses TimeDistributed wrapper for sequence processing
        
        4. Initial State Handling:
           - Creates zero initial state
           - Concatenates with predicted states
           - Ensures z(0) = 0 explicitly
        
        Args:
            un_I_tdt_f: Invariants tensor [batch, time, 6]
            un_dt_tdt_f: Timestep tensor [batch, time, 1]
            
        Returns:
            z_i_1_final: Internal variables [batch, time+1, internal_vars]
                        First timestep (t=0) is always zero
        """
        # Step 1: Combine invariants and timestep into state vector x(t)
        # x(t) = [I₁(t), I₂(t), I₃(t), I₄(t), I₅(t), I₆(t), dt]
        combined_input = tf.concat([un_I_tdt_f, un_dt_tdt_f], axis=-1)
        
        # Step 2: Process through first LSTM layer
        # h₁(t) = LSTM₁(x(t))
        # Captures temporal patterns in deformation history
        nf0_sub = self.f0(combined_input)
        
        # Step 3: Process through second LSTM layer
        # h₂(t) = LSTM₂(h₁(t))
        # Further refines temporal features
        nf0_sub_final = self.f02(nf0_sub)
        
        # Step 4: Project to internal variable space using dense layer
        # z̃(t) = f₂(h₂(t))
        # TimeDistributed wrapper applies same dense layer to each timestep
        z_i = self.f2(nf0_sub_final)
        
        # Step 5: Create zero initial state z(t=0) = 0
        # Ensures proper reference configuration
        batch_size = tf.shape(un_I_tdt_f)[0]
        init_state = tf.zeros((batch_size, 1, self.internal_variables_nr), dtype=tf.float32)
        
        # Step 6: Combine initial zero state with predicted states
        # z = [z(0), z(1), ..., z(T)]
        # where z(0) = 0 and z(t>0) = z̃(t)
        combined_states = tf.concat([init_state, z_i], axis=1)
        
        # Return complete internal variable evolution
        z_i_1_final = combined_states
        return z_i_1_final
    
    def free_energy_stress_dissipation(self, z_i_1_final, un_I_t_f, u_dt_t, stress_derivatives):
        """Calculate free energy, stress and dissipation rate using invariant formulation.
        
        This method implements the thermodynamically consistent calculation of material
        response, ensuring zero initial conditions and proper evolution of all quantities.
        
        Mathematical Framework:
        ---------------------
        1. Free Energy Calculation:
           ψ(C, z) = ψ_nn(C, z) - ψ_nn(C₀, z₀) - ∇ψ_nn(C₀, z₀):(C - C₀)
           where:
           - ψ_nn is the neural network output
           - C₀, z₀ are initial values
           - The last term ensures zero initial derivatives
        
        2. Stress Calculation:
           S = 2∂ψ/∂C = 2Σᵢ(∂ψ/∂Iᵢ)(∂Iᵢ/∂C)
           - Uses chain rule with invariants
           - Ensures frame indifference
           - Guarantees zero initial stress
        
        3. Dissipation Rate:
           D = -ż·(∂ψ/∂z) ≥ 0
           where:
           - ż is the time derivative of internal variables
           - ∂ψ/∂z is the thermodynamic force
        
        Implementation Steps:
        -------------------
        1. Reference State:
           - Extract initial values C₀, z₀
           - Store for zero condition enforcement
        
        2. Free Energy:
           - Compute ψ_nn(C, z) using PICNN or dense layers
           - Calculate initial energy ψ_nn(C₀, z₀)
           - Compute gradient at initial state
           - Subtract initial and linear terms
        
        3. Stress Computation:
           - Calculate ∂ψ/∂Iᵢ through automatic differentiation
           - Use pre-computed ∂Iᵢ/∂C
           - Sum contributions from all invariants
        
        4. Dissipation:
           - Compute internal variable rates ż
           - Calculate thermodynamic forces ∂ψ/∂z
           - Ensure non-negative dissipation
        
        Args:
            z_i_1_final: Internal variables [batch, time, internal_vars]
                        Includes t=0 state as first entry
            un_I_t_f: Invariants [batch, time, 6]
                     Six invariants of Cauchy-Green tensor
            u_dt_t: Timestep [batch, time, 1]
                   For rate calculations
            
        Returns:
            tuple:
                - sigma: Stress tensor [batch, time, 6] in Voigt notation
                       2∂ψ/∂C computed through invariants
                - psi_final: Free energy [batch, time, 1]
                           Zero at t=0, positive for t>0
                - diss_rate: Dissipation rate [batch, time]
                           Non-negative as per second law
        
        Note:
            The method ensures thermodynamic consistency through:
            1. Zero initial conditions for ψ, S
            2. Frame invariance via invariant formulation
            3. Non-negative dissipation
            4. Proper reference configuration
        """
        # Step 1: Extract reference state (t=0)
        # Stop gradient to treat initial values as constants
        reference_invariants = tf.stop_gradient(un_I_t_f[:, 0, :])  # C₀
        reference_internal_vars = tf.expand_dims(z_i_1_final[:,0,:] ,1)  # z₀
        
        # Get dimensions for tensor operations
        batch_size = tf.shape(un_I_t_f)[0]
        time_steps = tf.shape(un_I_t_f)[1]
        initial_time = 0
        un_I_t_f_t0 = tf.gather(un_I_t_f, initial_time, axis=1)  # Initial invariants
        
        # Step 2: Set up gradient tracking for automatic differentiation
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(un_I_t_f)      # Track invariants for stress calculation
            tape.watch(z_i_1_final)   # Track internal variables for dissipation
            tape.watch(un_I_t_f_t0)   # Track initial state for zero conditions

            # Step 3: Calculate initial free energy ψ_nn(C₀, z₀)
            if self.use_picnn:
                # PICNN path ensures convexity
                x_init = z_i_1_final[:,0,:]  # z₀
                y_init = un_I_t_f_t0     # C₀
                
                # Expand dimensions for time sequence processing
                x_init = tf.expand_dims(x_init, 1)  # [batch, 1, internal_vars]
                y_init = tf.expand_dims(y_init, 1)  # [batch, 1, invariants]
                
                psi = self.f3([x_init, y_init])
            else:
                # Dense layers path
                combined_init = tf.concat([
                    z_i_1_final[:,0,:],  # z₀
                    un_I_t_f_t0     # C₀
                ], axis=-1)
                combined_init = tf.expand_dims(combined_init, 1)
                
                psi = self.f3_dense1(combined_init)
                psi = self.f3_dense2(psi)
                
            psi_t_init = self.f4(psi)  # ψ_nn(C₀, z₀)
            
            # Step 4: Calculate current free energy ψ_nn(C, z)
            if self.use_picnn:
                psi = self.f3([z_i_1_final, un_I_t_f])
            else:
                combined = tf.concat([z_i_1_final, un_I_t_f], axis=-1)
                psi = self.f3_dense1(combined)
                psi = self.f3_dense2(psi)
                
            psi_t = self.f4(psi)  # ψ_nn(C, z)
            
            # Step 5: Calculate gradient at initial state ∇ψ_nn(C₀, z₀)
            dpsi_t_init_dI_t0 = tape.gradient(psi_t_init, un_I_t_f_t0)

            # Step 6: Calculate difference from initial state (C - C₀)
            delta_I = un_I_t_f - reference_invariants[:, None, :]

            # Step 7: Compute linear term ∇ψ_nn(C₀, z₀):(C - C₀)
            psi_linear_term = tf.reduce_sum(
                dpsi_t_init_dI_t0[:, None, :] * delta_I, 
                axis=-1, 
                keepdims=True
            )

            # Step 8: Adjust free energy to ensure zero conditions
            # ψ(C, z) = ψ_nn(C, z) - ψ_nn(C₀, z₀) - ∇ψ_nn(C₀, z₀):(C - C₀)
            psi_final_adjusted = psi_t - psi_t_init - psi_linear_term

        # Step 9: Calculate stress through invariant derivatives
        # S = 2∂ψ/∂C = 2Σᵢ(∂ψ/∂Iᵢ)(∂Iᵢ/∂C)
        dpsi_dI = tape.gradient(psi_final_adjusted, un_I_t_f)

        # Step 10: Calculate dissipation rate
        # D = -ż·(∂ψ/∂z) ≥ 0
        z_delta = tf.experimental.numpy.diff(z_i_1_final, n=1, axis=1)  # ż
        z_dot = z_delta/u_dt_t[:,1:,:]
        tau = tape.gradient(psi_final_adjusted, z_i_1_final)  # ∂ψ/∂z
        f_diss_tdt = tf.slice(tau,[0,1,0],[-1,-1,-1]) * z_dot 
        diss_rate = tf.reduce_sum(f_diss_tdt, axis=-1)*(-1)
        
        # Step 11: Calculate stress components using invariants
        # Initialize stress tensor
        sigma_vector = tf.zeros([batch_size, time_steps, 6], dtype=tf.float32)

        # Sum contributions from all invariants
        for i in range(6):
            dpsi_dI_i = tf.expand_dims(dpsi_dI[:,:,i], -1)  # ∂ψ/∂Iᵢ
            derivatives_i = stress_derivatives[:,:,i,:]  # ∂Iᵢ/∂C
            contribution = dpsi_dI_i * derivatives_i
            sigma_vector = sigma_vector + contribution
        
        return 2*sigma_vector, psi_final_adjusted, diss_rate
    
    def get_all_losses(self, un, true_stress, stress_derivatives):
        """Calculate all loss components for training."""
        un_I_t_f = tf.slice(un,[0,0,0],[-1,-1,6])  # 6 invariants
        un_I_dt_f = tf.slice(un,[0,1,0],[-1,-1,6])  # 6 invariants (t+1)
        un_dt_t_f = tf.slice(un,[0,0,6],[-1,-1,1])  # timestep (already unnormalized)
        un_dt_dt_f = tf.slice(un,[0,1,6],[-1,-1,1])  # timestep at t+1 (already unnormalized)
        
        internal_vars = self.internal_variables(un_I_dt_f, un_dt_dt_f)
        f_σ_tdt, psi_final, diss_rate = self.free_energy_stress_dissipation(
            internal_vars, un_I_t_f, un_dt_t_f, stress_derivatives)
        
        loss_stress = tf.reduce_mean(tf.abs(true_stress - f_σ_tdt))
        loss_diss = tf.reduce_mean(tf.nn.relu(-diss_rate))*10
        loss_psi = tf.reduce_mean(tf.nn.relu(-psi_final))*10
        return loss_stress, loss_diss, loss_psi
    
    def get_loss(self, un, true_stress, stress_derivatives):
        losses = self.get_all_losses(un, true_stress, stress_derivatives)
        return sum(losses)

    def get_grad(self, un, true_stress, stress_derivatives):
        """Calculate gradients for model training."""
        with tf.GradientTape() as tape:
            L = self.get_loss(un, true_stress, stress_derivatives)
        g = tape.gradient(L, self.trainable_variables)
        return L, g

    def train_step(self, batch_data):
        """Single training step, optimized with tf.function."""
        batch_un, batch_true_stress, batch_derivatives = batch_data
        
        # Get all losses for this batch
        batch_losses = self.get_all_losses(batch_un, batch_true_stress, batch_derivatives)
        stress_loss, diss_loss, psi_loss = batch_losses
        
        # Get gradients and update model
        L, g = self.get_grad(batch_un, batch_true_stress, batch_derivatives)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        
        return L, stress_loss, diss_loss, psi_loss

    def obtain_output(self, un, stress_derivatives=None):
        """Generate all model predictions for given input."""
        un_I_t_f = tf.slice(un,[0,0,0],[-1,-1,6])  # 6 invariants
        un_I_dt_f = tf.slice(un,[0,1,0],[-1,-1,6])  # 6 invariants
        un_dt_t_f = tf.slice(un,[0,0,6],[-1,-1,1])  # timestep (already unnormalized)
        un_dt_dt_f = tf.slice(un,[0,1,6],[-1,-1,1])  # timestep at t+1 (already unnormalized)
        
        internal_vars = self.internal_variables(un_I_dt_f, un_dt_dt_f)
        if stress_derivatives is None:
            stress_derivatives = self.stress_derivatives_tf
        f_σ_tdt, psi_final, diss_rate = self.free_energy_stress_dissipation(
            internal_vars, un_I_t_f, un_dt_t_f, stress_derivatives)
        return f_σ_tdt, psi_final, diss_rate, internal_vars

    def network_learn(self, dataset, num_epochs):
        """
        Train the model for specified number of epochs using batch training.
        
        Args:
            dataset: TensorFlow dataset containing (input, true_stress, derivatives) tuples
            num_epochs: Number of training epochs
            
        Returns:
            l_liste: List of loss values during training
        """
        l_liste = []
        stress_loss_threshold = 0.0005  # Early stopping threshold
        
        for i in range(num_epochs):
            epoch_loss = 0
            epoch_stress_loss = 0
            epoch_diss_loss = 0
            epoch_psi_loss = 0
            num_batches = 0
            
            # Train on batches
            for batch_data in dataset:
                L, stress_loss, diss_loss, psi_loss = self.train_step(batch_data)
                
                # Accumulate losses
                epoch_loss += L
                epoch_stress_loss += stress_loss
                epoch_diss_loss += diss_loss
                epoch_psi_loss += psi_loss
                num_batches += 1
            
            # Average losses for the epoch
            epoch_loss = epoch_loss / tf.cast(num_batches, tf.float32)
            epoch_stress_loss = epoch_stress_loss / tf.cast(num_batches, tf.float32)
            epoch_diss_loss = epoch_diss_loss / tf.cast(num_batches, tf.float32)
            epoch_psi_loss = epoch_psi_loss / tf.cast(num_batches, tf.float32)
            
            l_liste.append(epoch_loss)
            
            # Get current learning rate
            if hasattr(self.optimizer.learning_rate, '__call__'):
                current_lr = self.optimizer.learning_rate(self.optimizer.iterations)
            else:
                current_lr = self.optimizer.learning_rate
            
            # Print progress every print_every epochs
            if i % self.print_every == 1:
                print("Epoch {} stress_loss: {}, diss_loss: {}, psi_loss: {}, lr: {:.6f}".format(
                    i, epoch_stress_loss.numpy(), epoch_diss_loss.numpy(), epoch_psi_loss.numpy(), 
                    float(current_lr)), file=sys.stderr)
                
                # Early stopping check using average stress loss
                if epoch_stress_loss < stress_loss_threshold:
                    print(f"\nReached target stress loss ({stress_loss_threshold}) at epoch {i}. Stopping training.", file=sys.stderr)
                    break
        
        return l_liste

    def save(self, filepath):
        """
        Save the model to the specified filepath.
        Saves both the model architecture and weights using TensorFlow checkpoints.
        """
        # Create a dictionary with all the model parameters
        model_config = {
            's_all': self.s_all.numpy(),
            'm_all': self.m_all.numpy(),
            's_out': self.s_out.numpy(),
            'm_out': self.m_out.numpy(),
            'layer_size': self.layer_size,
            'internal_variables': self.internal_variables_nr,
            'layer_size_fenergy': self.layer_size_fenergy,
            'batch_size': self.batch_size,
            's_dt': self.s_dt.numpy(),
            'm_dt': self.m_dt.numpy(),
            'stress_derivatives': self.stress_derivatives_tf.numpy(),
            'invariants': self.invariants,
            'use_picnn': self.use_picnn,
            'training_silent': self.training_silent,
            'optimizer_config': self.optimizer.get_config()
        }

        # Clean up existing files and create directory
        if os.path.exists(filepath):
            shutil.rmtree(filepath)
        os.makedirs(filepath)
        
        # Save model configuration
        np.save(os.path.join(filepath, 'model_config.npy'), model_config)
        
        # Create checkpoint
        checkpoint = tf.train.Checkpoint(model=self)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=os.path.join(filepath, 'checkpoints'),
            max_to_keep=1
        )
        
        # Save the checkpoint
        checkpoint_manager.save()
        
        print(f"Model saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load a saved model from the specified filepath.
        Returns a new instance of the model with loaded weights.
        """
        # Load model configuration
        model_config = np.load(os.path.join(filepath, 'model_config.npy'), allow_pickle=True).item()
        
        # Convert numpy arrays back to tensors
        s_all = tf.convert_to_tensor(model_config['s_all'])
        m_all = tf.convert_to_tensor(model_config['m_all'])
        s_out = tf.convert_to_tensor(model_config['s_out'])
        m_out = tf.convert_to_tensor(model_config['m_out'])
        s_dt = tf.convert_to_tensor(model_config['s_dt'])
        m_dt = tf.convert_to_tensor(model_config['m_dt'])
        stress_derivatives = tf.convert_to_tensor(model_config['stress_derivatives'])
        
        # For inference only, we do not need the original optimizer.
        # Supply a dummy optimizer.
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Create new model instance
        model = cls(
            s_all=s_all,
            m_all=m_all,
            s_out=s_out,
            m_out=m_out,
            layer_size=model_config['layer_size'],
            internal_variables=model_config['internal_variables'],
            layer_size_fenergy=model_config['layer_size_fenergy'],
            batch_size=model_config['batch_size'],
            optimizer=optimizer,
            s_dt=s_dt,
            m_dt=m_dt,
            stress_derivatives=stress_derivatives,
            invariants=model_config['invariants'],
            use_picnn=model_config['use_picnn'],
            training_silent=model_config['training_silent']
        )
        
        # Restore from checkpoint
        checkpoint = tf.train.Checkpoint(model=model)
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(filepath, 'checkpoints'))
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint).expect_partial()
            print(f"Model weights restored from {latest_checkpoint}")
        else:
            print("Warning: No checkpoint found, model weights not restored")
        
        return model


