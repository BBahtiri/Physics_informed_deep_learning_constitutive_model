import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def load_excel_data(file_path):
    """Load strain and stress data from Excel file.
    
    Args:
        file_path: Path to Excel file containing strain and stress data
        
    Returns:
        tuple: (strain_data, sigma_data, time_data)
            - strain_data: Green-Lagrange strain tensor [batch, time, 6]
            - sigma_data: Second Piola-Kirchhoff stress tensor [batch, time, 6]
            - time_data: Time steps [batch, time]
    """
    # Read all sheets from the Excel file
    excel_file = pd.ExcelFile(file_path)
    
    # Extract components
    strain_cols = ['E11', 'E12', 'E13', 'E22', 'E23', 'E33']
    stress_cols = ['S11', 'S12', 'S13', 'S22', 'S23', 'S33']
    time_col = 'DT'
    
    # Initialize arrays
    nr_sheets = len(excel_file.sheet_names)
    first_sheet = pd.read_excel(file_path, sheet_name=excel_file.sheet_names[0])
    timesteps = len(first_sheet)
    
    # Initialize arrays for strain tensor
    strain_data = np.zeros((nr_sheets, timesteps, len(strain_cols)), dtype=np.float32)
    sigma_data = np.zeros((nr_sheets, timesteps, len(stress_cols)), dtype=np.float32)
    time_data = np.zeros((nr_sheets, timesteps), dtype=np.float32)
    
    # Process each sheet
    for idx, sheet_name in enumerate(excel_file.sheet_names):
        # Read the specific sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Set small values to zero (both strains and stresses)
        df[strain_cols] = df[strain_cols].where(abs(df[strain_cols]) > 1e-9, 0.0)
        df[stress_cols] = df[stress_cols].where(abs(df[stress_cols]) > 1e-9, 0.0)
        
        # Store strain components directly in Voigt notation
        strain_data[idx, :, 0] = df[strain_cols[0]].values  # E11
        strain_data[idx, :, 1] = df[strain_cols[1]].values  # E12
        strain_data[idx, :, 2] = df[strain_cols[2]].values  # E13
        strain_data[idx, :, 3] = df[strain_cols[3]].values  # E22
        strain_data[idx, :, 4] = df[strain_cols[4]].values  # E23
        strain_data[idx, :, 5] = df[strain_cols[5]].values  # E33
        
        # Store stress and time data
        sigma_data[idx] = df[stress_cols].values
        time_data[idx] = df[time_col].values
    
    return strain_data, sigma_data, time_data


def calculate_invariants(strain):
    """Calculate the six invariants of the strain tensor.
    
    The invariants characterize the deformation state of the material:
    
    I₁ = tr(ε) = ε₁₁ + ε₂₂ + ε₃₃
        First invariant: trace of strain tensor
        
    I₂ = tr(ε²)
        Second invariant: trace of squared strain tensor
        
    I₃ = tr(Aε)
        Third invariant: trace of strain tensor with structural tensor
        
    I₄ = tr(A(ε)²)
        Fourth invariant: trace of squared strain with structural tensor
        
    I₅ = tr(ε³)
        Fifth invariant: trace of cubed strain tensor
        
    I₆ = -2√det(2ε + I)
        Sixth invariant: related to volume change
    
    Args:
        strain: Green-Lagrange strain tensor in Voigt notation [E11, E12, E13, E22, E23, E33]
               Shape: [batch_size, timesteps, 6]
    
    Returns:
        tf.Tensor: Tensor of shape [batch_size, timesteps, 6] containing all invariants
    """
    batch_size, timesteps, _ = strain.shape
    invariants = np.zeros((batch_size, timesteps, 6), dtype=np.float32)
    
    # Define structural tensor a₀ = [1,0,0]
    a0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    A = np.outer(a0, a0)  # Structural tensor A = a₀ ⊗ a₀
    
    for b in range(batch_size):
        for t in range(timesteps):
            # Reshape 6D vector to 3x3 symmetric matrix
            E = np.zeros((3,3), dtype=np.float32)
            E[0,0] = strain[b,t,0]  # E11
            E[0,1] = E[1,0] = strain[b,t,1]  # E12
            E[0,2] = E[2,0] = strain[b,t,2]  # E13
            E[1,1] = strain[b,t,3]  # E22
            E[1,2] = E[2,1] = strain[b,t,4]  # E23
            E[2,2] = strain[b,t,5]  # E33
            
            # Calculate invariants using float32
            # I₁ = tr(ε)
            I1 = np.float32(np.trace(E))
            
            # I₂ = tr(ε²)
            E2 = np.matmul(E, E)
            I2 = np.float32(np.trace(E2))
            
            # I₃ = tr(Aε)
            AE = np.matmul(A, E)
            I3 = np.float32(np.trace(AE))
            
            # I₄ = tr(A(ε)²)
            E2 = np.matmul(E, E)  # ε²
            AE2 = np.matmul(A, E2)  # A(ε²)
            I4 = np.float32(np.trace(AE2))
            
            # I₅ = tr(ε³)
            E3 = np.matmul(E2, E)  # ε³
            I5 = np.float32(np.trace(E3))
            
            # I₆ = -2√det(2ε + I) (keep as before)
            C = 2 * E + np.eye(3, dtype=np.float32)
            I6 = np.float32(-2.0 * np.sqrt(np.linalg.det(C)))
            
            invariants[b,t] = [I1, I2, I3, I4, I5, I6]
    
    return tf.convert_to_tensor(invariants, dtype=tf.float32)


def calculate_stress_derivatives(strain):
    """Calculate the derivatives of invariants with respect to the strain tensor.
    
    The derivatives are:
    ∂I₁/∂ε = 1 (identity tensor)
        First invariant derivative
        
    ∂I₂/∂ε = 2ε
        Second invariant derivative
        
    ∂I₃/∂ε = A
        Third invariant derivative
        
    ∂I₄/∂ε = Aε + εA
        Fourth invariant derivative
        
    ∂I₅/∂ε = 3(ε)²
        Fifth invariant derivative
        
    ∂I₆/∂ε = -J·C⁻¹ (kept as before)
        Sixth invariant derivative
    
    Args:
        strain: Green-Lagrange strain tensor in Voigt notation [E11, E12, E13, E22, E23, E33]
               Shape: [batch_size, timesteps, 6]
    
    Returns:
        tf.Tensor: Tensor of shape [batch_size, timesteps, 6, 6] containing derivatives
                  [batch, time, invariant, component]
    """
    batch_size, timesteps, _ = strain.shape
    derivatives = np.zeros((batch_size, timesteps, 6, 6), dtype=np.float32)
    
    # Define structural tensor a₀ = [1,0,0]
    a0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    A = np.outer(a0, a0)  # Structural tensor A = a₀ ⊗ a₀
    
    # Identity matrix
    I_mat = np.eye(3, dtype=np.float32)
    
    for b in range(batch_size):
        for t in range(timesteps):
            # Construct strain matrix
            E = np.zeros((3,3), dtype=np.float32)
            E[0,0] = strain[b,t,0]  # E11
            E[0,1] = E[1,0] = strain[b,t,1]  # E12
            E[0,2] = E[2,0] = strain[b,t,2]  # E13
            E[1,1] = strain[b,t,3]  # E22
            E[1,2] = E[2,1] = strain[b,t,4]  # E23
            E[2,2] = strain[b,t,5]  # E33
            
            # Calculate derivatives using float32
            # ∂I₁/∂ε = 1
            dI1_dE = I_mat
            
            # ∂I₂/∂ε = 2ε
            dI2_dE = 2 * E
            
            # ∂I₃/∂ε = A
            dI3_dE = A
            
            # ∂I₄/∂ε = Aε + εA
            dI4_dE = np.matmul(A, E) + np.matmul(E, A)
            
            # ∂I₅/∂ε = 3(ε)²
            E2 = np.matmul(E, E)
            dI5_dE = 3 * E2
            
            # ∂I₆/∂ε (keep as before)
            C = 2 * E + I_mat
            J = np.float32(np.sqrt(np.linalg.det(C)))
            C_inv = np.linalg.inv(C)
            dI6_dE = -2 * J * C_inv
            
            # Store derivatives in Voigt notation
            derivatives[b,t,0] = [dI1_dE[0,0], dI1_dE[0,1], dI1_dE[0,2],
                                dI1_dE[1,1], dI1_dE[1,2], dI1_dE[2,2]]
            
            derivatives[b,t,1] = [dI2_dE[0,0], dI2_dE[0,1], dI2_dE[0,2],
                                dI2_dE[1,1], dI2_dE[1,2], dI2_dE[2,2]]
            
            derivatives[b,t,2] = [dI3_dE[0,0], dI3_dE[0,1], dI3_dE[0,2],
                                dI3_dE[1,1], dI3_dE[1,2], dI3_dE[2,2]]
            
            derivatives[b,t,3] = [dI4_dE[0,0], dI4_dE[0,1], dI4_dE[0,2],
                                dI4_dE[1,1], dI4_dE[1,2], dI4_dE[2,2]]
            
            derivatives[b,t,4] = [dI5_dE[0,0], dI5_dE[0,1], dI5_dE[0,2],
                                dI5_dE[1,1], dI5_dE[1,2], dI5_dE[2,2]]
            
            derivatives[b,t,5] = [dI6_dE[0,0], dI6_dE[0,1], dI6_dE[0,2],
                                dI6_dE[1,1], dI6_dE[1,2], dI6_dE[2,2]]
    
    return tf.convert_to_tensor(derivatives, dtype=tf.float32)


def plot_detailed_data(strain, sigma):
    """Plot detailed strain and stress data."""
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    # Create figure for strain components
    fig_strain, axes_strain = plt.subplots(2, 3, figsize=(15, 10))
    axes_strain = axes_strain.flatten()
    
    for i in range(6):
        for j in range(strain.shape[0]):
            axes_strain[i].plot(strain[j, :, i].numpy())
        axes_strain[i].set_title(f'Strain Component {i+1}')
        axes_strain[i].grid(True)
    
    plt.tight_layout()
    # Show plot interactively
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure plot renders
    
    # Also save the plot
    fig_strain.savefig('plots/strain_components.png')
    plt.close(fig_strain)
    
    # Create figure for stress components
    fig_stress, axes_stress = plt.subplots(2, 3, figsize=(15, 10))
    axes_stress = axes_stress.flatten()
    
    for i in range(6):
        for j in range(sigma.shape[0]):
            axes_stress[i].plot(sigma[j, :, i].numpy())
        axes_stress[i].set_title(f'Stress Component {i+1}')
        axes_stress[i].grid(True)
    
    plt.tight_layout()
    # Show plot interactively
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure plot renders
    
    # Also save the plot
    fig_stress.savefig('plots/stress_components.png')
    plt.close(fig_stress)
    
    print("Plots displayed and saved as 'strain_components.png' and 'stress_components.png' in the plots directory")
    plt.close('all')  # Close any remaining plots


def plot_normalized_data(normalized_train_x, normalized_train_y):
    """Create visualization of normalized invariants and stresses.
    
    Generates a 2x7 grid of subplots:
    - Top row: 6 invariants (I₁-I₆) and timestep (dt)
    - Bottom row: 6 stress components (σ₁₁-σ₃₃)
    All values are normalized to comparable ranges.
    
    Args:
        normalized_train_x: Normalized input tensor [batch, time, 7]
        normalized_train_y: Normalized stress tensor [batch, time, 6]
    
    Saves:
        'norm_data.png': Figure showing normalized quantities over time
    """
    # Create two rows of subplots: one for invariants, one for stresses
    fig, axs = plt.subplots(2, 7, figsize=(28, 8))  # Changed from 6 to 7 columns
    
    # Convert tensors to numpy if needed
    train_x_np = normalized_train_x.numpy() if isinstance(normalized_train_x, tf.Tensor) else normalized_train_x
    train_y_np = normalized_train_y.numpy() if isinstance(normalized_train_y, tf.Tensor) else normalized_train_y
    
    # Plot normalized invariants (first row)
    invariant_names = ['I₁', 'I₂', 'I₃', 'I₄', 'I₅', 'I₆', 'dt']  # Added I₆
    for i in range(7):  # Changed from 6 to 7 (6 invariants + timestep)
        ax = axs[0, i]
        for dataset_idx in range(train_x_np.shape[0]):
            ax.plot(train_x_np[dataset_idx, :, i], 
                   label=f'Dataset {dataset_idx+1}' if i == 0 else None)
        ax.set_title(f'Normalized {invariant_names[i]}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Normalized Value')
        ax.grid(True)
        if i == 0:
            ax.legend()
    
    # Plot normalized stresses (second row)
    stress_components = ['σ₁₁', 'σ₁₂', 'σ₁₃', 'σ₂₂', 'σ₂₃', 'σ₃₃']
    for i in range(6):
        ax = axs[1, i]
        for dataset_idx in range(train_y_np.shape[0]):
            ax.plot(train_y_np[dataset_idx, :, i])
        ax.set_title(f'Normalized {stress_components[i]}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Normalized Value')
        ax.grid(True)
    
    # Hide the last subplot in the second row (since we only have 6 stress components)
    axs[1, 6].set_visible(False)
    
    plt.suptitle('Normalized Input (Invariants) and Output (Stresses)', fontsize=16)
    plt.tight_layout()
    
    # Show plot interactively
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure plot renders
    
    # Save the plot with shorter path
    plt.savefig('plots/norm_data.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    plt.close('all')  # Close any remaining plots


def plot_normalized_derivatives(normalized_derivatives):
    """Create visualization of normalized stress derivative components.
    
    Generates a 6x6 grid of subplots showing the derivatives of each
    invariant with respect to each strain component:
    ∂Iᵢ/∂εⱼ where i,j ∈ {1,...,6}
    
    Args:
        normalized_derivatives: Normalized derivative tensor [batch, time, 6, 6]
    
    Saves:
        'norm_deriv.png': Figure showing normalized derivatives over time
    """
    # Create subplots: 6 rows (one for each invariant) x 6 columns (stress components)
    fig, axs = plt.subplots(6, 6, figsize=(24, 20))
    
    # Convert tensor to numpy if needed
    derivatives_np = normalized_derivatives.numpy() if isinstance(normalized_derivatives, tf.Tensor) else normalized_derivatives
    
    # Labels for better visualization
    invariant_names = ['∂I₁/∂ε', '∂I₂/∂ε', '∂I₃/∂ε', '∂I₄/∂ε', '∂I₅/∂ε', '∂I₆/∂ε']
    stress_components = ['11', '12', '13', '22', '23', '33']
    
    # Plot each derivative component
    for i in range(6):  # 6 invariants
        for j in range(6):  # 6 stress components
            ax = axs[i, j]
            # Plot for each dataset
            for dataset_idx in range(derivatives_np.shape[0]):
                ax.plot(derivatives_np[dataset_idx, :, i, j],
                       label=f'Dataset {dataset_idx+1}' if j == 0 else None)
            
            # Set titles and labels
            if i == 0:
                ax.set_title(f'Component {stress_components[j]}')
            if j == 0:
                ax.set_ylabel(f'{invariant_names[i]}')
            ax.grid(True)
            if j == 0:
                ax.legend()
            
            # Set y-axis limits to [-1.1, 1.1] for normalized values
            ax.set_ylim([-1.1, 1.1])
    
    plt.suptitle('Normalized Stress Derivatives', fontsize=16)
    plt.tight_layout()
    
    # Show plot interactively
    plt.show(block=False)
    plt.pause(0.1)  # Small pause to ensure plot renders
    
    # Save the plot with shorter path
    plt.savefig('plots/norm_deriv.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    plt.close('all')  # Close any remaining plots


def plot_predicted_vs_true(epsilon, true_stress, predicted_stress, s_y, m_y):
    """Create comparative visualization of predicted vs true stress-strain behavior.
    
    Generates a 2x3 grid of subplots comparing the model predictions
    with ground truth data for each stress component. Includes MSE calculation
    and proper denormalization of predicted stresses.
    
    Args:
        epsilon: Strain tensor [batch, time, 6]
        true_stress: True stress tensor [batch, time, 6]
        predicted_stress: Predicted stress tensor (normalized) [batch, time, 6]
        s_y: Stress scaling factors [6]
        m_y: Stress mean values [6]
    
    Saves:
        'pred_vs_true.png': Figure comparing predicted and true stresses
    """
    n_components = 6  # Number of strain/stress components
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    axs = axs.ravel()
    
    # Component names for better labeling
    component_names = ['11', '12', '13', '22', '23', '33']
    
    # Convert tensors to numpy if needed
    epsilon_np = epsilon.numpy() if isinstance(epsilon, tf.Tensor) else epsilon
    true_stress_np = true_stress.numpy() if isinstance(true_stress, tf.Tensor) else true_stress
    predicted_stress_np = predicted_stress.numpy() if isinstance(predicted_stress, tf.Tensor) else predicted_stress
    
    # Unnormalize predicted stress by multiplying with scaling factor
    predicted_stress_np = predicted_stress_np * s_y
    mse = np.mean((predicted_stress_np - true_stress_np)**2)
    
    # Plot each component
    for i in range(n_components):
        ax = axs[i]
        
        # Plot for each dataset
        for dataset_idx in range(epsilon_np.shape[0]):
            # True stress-strain curve
            ax.plot(epsilon_np[dataset_idx, :, i], true_stress_np[dataset_idx, :, i], 
                   'b-', label=f'True (Dataset {dataset_idx+1})' if dataset_idx == 0 else None,
                   alpha=0.7)
            
            # Predicted stress-strain curve
            ax.plot(epsilon_np[dataset_idx, :, i], predicted_stress_np[dataset_idx, :, i], 
                   'r--', label=f'Predicted (Dataset {dataset_idx+1})' if dataset_idx == 0 else None,
                   alpha=0.7)
        
        ax.set_xlabel(f'Strain ε{component_names[i]}')
        ax.set_ylabel(f'Stress σ{component_names[i]}')
        ax.set_title(f'Component {component_names[i]}')
        ax.grid(True)
        if i == 0:  # Only add legend to first subplot
            ax.legend()
    
    plt.suptitle(f'Predicted vs True Stress-Strain Behavior (MSE: {mse:.6f})', fontsize=16)
    plt.tight_layout()
    
    # Save the plot with shorter path
    plt.savefig('plots/pred_vs_true.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def normalize_invariants(data):
    """Normalize the invariants of the Cauchy-Green tensor to the range [-1, 1].
    
    The normalization is performed using min-max scaling:
    x_norm = 2 * (x - min(x))/(max(x) - min(x)) - 1
    
    The timestep component is kept unchanged.
    
    Args:
        data: Tensor of shape [batch_size, timesteps, 7] containing 6 invariants and timestep
    
    Returns:
        tuple: (normalized_data, scale_factors, mean_values)
            - normalized_data: Normalized tensor of same shape as input
            - scale_factors: Scaling factors for each component
            - mean_values: Mean values for each component
    """
    # Convert input to float32 if needed
    data = tf.cast(data, tf.float32)
    
    # Normalization of input (invariants, not timestep)
    max_values_x = tf.reduce_max(data[:,:,:6], axis=[0,1])
    min_values_x = tf.reduce_min(data[:,:,:6], axis=[0,1])
    s_x = tf.concat([(max_values_x - min_values_x), tf.ones(1, dtype=tf.float32)], axis=0)
    m_x = tf.concat([min_values_x, tf.zeros(1, dtype=tf.float32)], axis=0)

    # Normalize invariants to [-1, 1] range, keeping original values where normalization gives NaN
    normalized_train_x = tf.concat([
        (data[:,:,:6] - m_x[:6]) / s_x[:6] * 2 - 1,  # Normalize invariants to [-1, 1]
        data[:,:,6:]  # Keep timestep as is
    ], axis=-1)

    # Replace NaN values with original unnormalized values
    normalized_train_x = tf.where(
        tf.math.is_finite(normalized_train_x),
        normalized_train_x,
        data
    )

    return normalized_train_x, s_x, m_x

def normalize_stresses(data):
    """Normalize the stress components by their maximum absolute values.
    
    The normalization preserves the sign of the stress components:
    σ_norm = σ / max(|σ|)
    
    This scaling ensures that all stress components are in a similar range
    while maintaining their relative magnitudes and signs.
    
    Args:
        data: Stress tensor of shape [batch_size, timesteps, 6]
    
    Returns:
        tuple: (normalized_data, scale_factors, mean_values)
            - normalized_data: Normalized stress tensor
            - scale_factors: Maximum absolute values for each component
            - mean_values: Zero values (stress normalization is centered at zero)
    """
    # Convert input to float32 if needed
    data = tf.cast(data, tf.float32)
    
    # Normalization of output (stresses)
    max_abs_values_y = tf.reduce_max(tf.abs(data), axis=[0,1])
    s_y = max_abs_values_y
    m_y = tf.zeros_like(max_abs_values_y)

    # Normalize stresses by dividing by max absolute values
    normalized_train_y = data / s_y[None, None, :]

    # Replace NaN values with original unnormalized values
    normalized_train_y = tf.where(
        tf.math.is_finite(normalized_train_y),
        normalized_train_y,
        data
    )

    return normalized_train_y, s_y, m_y


def normalize_stress_derivatives(data):
    """Normalize the stress derivative components to the range [-1, 1].
    
    The normalization is performed using min-max scaling:
    x_norm = 2 * (x - min(x))/(max(x) - min(x)) - 1
    
    This ensures that all derivative components are in a comparable range
    while preserving their relative relationships.
    
    Args:
        data: Derivative tensor of shape [batch_size, timesteps, 6, 6]
              [batch, time, invariant, component]
    
    Returns:
        tuple: (normalized_data, scale_factors, mean_values)
            - normalized_data: Normalized derivative tensor
            - scale_factors: Range (max - min) for each component
            - mean_values: Minimum values for each component
    """
    # Convert input to float32 if needed
    data = tf.cast(data, tf.float32)
    
    # Normalization of stress derivatives
    max_values_deriv = tf.reduce_max(data, axis=[0, 1])
    min_values_deriv = tf.reduce_min(data, axis=[0, 1])
    s_deriv = (max_values_deriv - min_values_deriv)
    m_deriv = min_values_deriv
    normalized_stress_derivatives = (data - m_deriv[None, None, :, :]) / s_deriv[None, None, :, :] * 2 - 1
    # Replace NaN values with original unnormalized values
    normalized_stress_derivatives = tf.where(
        tf.math.is_finite(normalized_stress_derivatives),
        normalized_stress_derivatives,
        data
    )

    return normalized_stress_derivatives, s_deriv, m_deriv

def plot_unnormalized_data(train_x, train_y):
    """Create visualization of unnormalized invariants and stresses.
    
    Generates a 2x7 grid of subplots:
    - Top row: 6 invariants (I₁-I₆) and timestep (dt)
    - Bottom row: 6 stress components (σ₁₁-σ₃₃)
    Shows the actual physical values before normalization.
    
    Args:
        train_x: Input tensor [batch, time, 7] containing invariants and timestep
        train_y: Stress tensor [batch, time, 6]
    
    Saves:
        'unnorm_data.png': Figure showing unnormalized quantities over time
    """
    # Create two rows of subplots: one for invariants, one for stresses
    fig, axs = plt.subplots(2, 7, figsize=(28, 8))
    
    # Convert tensors to numpy if needed
    train_x_np = train_x.numpy() if isinstance(train_x, tf.Tensor) else train_x
    train_y_np = train_y.numpy() if isinstance(train_y, tf.Tensor) else train_y
    
    # Plot unnormalized invariants (first row)
    invariant_names = ['I₁', 'I₂', 'I₃', 'I₄', 'I₅', 'I₆', 'dt']
    for i in range(7):
        ax = axs[0, i]
        for dataset_idx in range(train_x_np.shape[0]):
            ax.plot(train_x_np[dataset_idx, :, i], 
                   label=f'Dataset {dataset_idx+1}' if i == 0 else None)
        ax.set_title(f'{invariant_names[i]}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.grid(True)
        if i == 0:
            ax.legend()
    
    # Plot unnormalized stresses (second row)
    stress_components = ['σ₁₁', 'σ₁₂', 'σ₁₃', 'σ₂₂', 'σ₂₃', 'σ₃₃']
    for i in range(6):
        ax = axs[1, i]
        for dataset_idx in range(train_y_np.shape[0]):
            ax.plot(train_y_np[dataset_idx, :, i])
        ax.set_title(f'{stress_components[i]}')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    # Hide the last subplot in the second row (since we only have 6 stress components)
    axs[1, 6].set_visible(False)
    
    plt.suptitle('Unnormalized Input (Invariants) and Output (Stresses)', fontsize=16)
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    
    # Save the plot
    plt.savefig('plots/unnorm_data.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_unnormalized_derivatives(derivatives):
    """Create visualization of unnormalized stress derivative components.
    
    Generates a 6x6 grid of subplots showing the derivatives of each
    invariant with respect to each strain component:
    ∂Iᵢ/∂εⱼ where i,j ∈ {1,...,6}
    Shows the actual physical values before normalization.
    
    Args:
        derivatives: Derivative tensor [batch, time, 6, 6]
    
    Saves:
        'unnorm_deriv.png': Figure showing unnormalized derivatives over time
    """
    # Create subplots: 6 rows (one for each invariant) x 6 columns (stress components)
    fig, axs = plt.subplots(6, 6, figsize=(24, 20))
    
    # Convert tensor to numpy if needed
    derivatives_np = derivatives.numpy() if isinstance(derivatives, tf.Tensor) else derivatives
    
    # Labels for better visualization
    invariant_names = ['∂I₁/∂ε', '∂I₂/∂ε', '∂I₃/∂ε', '∂I₄/∂ε', '∂I₅/∂ε', '∂I₆/∂ε']
    stress_components = ['11', '12', '13', '22', '23', '33']
    
    # Plot each derivative component
    for i in range(6):  # 6 invariants
        for j in range(6):  # 6 stress components
            ax = axs[i, j]
            # Plot for each dataset
            for dataset_idx in range(derivatives_np.shape[0]):
                ax.plot(derivatives_np[dataset_idx, :, i, j],
                       label=f'Dataset {dataset_idx+1}' if j == 0 else None)
            
            # Set titles and labels
            if i == 0:
                ax.set_title(f'Component {stress_components[j]}')
            if j == 0:
                ax.set_ylabel(f'{invariant_names[i]}')
            ax.grid(True)
            if j == 0:
                ax.legend()
    
    plt.suptitle('Unnormalized Stress Derivatives', fontsize=16)
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    
    # Save the plot
    plt.savefig('plots/unnorm_deriv.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
