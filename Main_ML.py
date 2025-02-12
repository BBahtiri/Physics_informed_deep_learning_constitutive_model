# -*- coding: utf-8 -*-
"""Physics-Informed Deep Learning (PIDL) for Material Modeling

This module implements a PIDL framework for learning constitutive material behavior
from experimental data. The model combines physical constraints with data-driven learning
to predict material responses under various loading conditions.

Key Components:
    1. Data Processing:
        - Loads experimental data from Excel files
        - Calculates Cauchy-Green deformation tensor invariants
        - Computes stress derivatives for constitutive modeling
        
    2. Neural Network Architecture:
        - LSTM layers for capturing history-dependent behavior
        - Dense layers for internal variable prediction
        - PICNN (Partially Input Convex Neural Network) for free energy prediction
        
    3. Physics-Informed Constraints:
        - Thermodynamic consistency through free energy formulation
        - Non-negative dissipation rate
        - Proper invariant-based formulation
                     
    4. Training and Validation:
        - Custom loss functions incorporating physical constraints
        - Adaptive learning rate scheduling
        - Comprehensive visualization of results

The model is designed to learn complex material behaviors while ensuring
physical consistency and interpretability of the predictions.

Author: bahtiri
Created: Wed Aug 23 08:46:13 2023
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib
# Use interactive backend for matplotlib
#matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
#plt.ion()  # Turn on interactive mode
from math import pi, cos, sin, sqrt
import pandas as pd
from pathlib import Path
from utils.PIDL import PIDL
from utils.scipy_loss import scipy_function_factory
from utils.misc import *
from Trained_Model import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    import nvidia.cudnn
    cudnn_path = Path(nvidia.cudnn.__file__).parent
    cudnn_lib_path = cudnn_path / "lib"
    os.environ["LD_LIBRARY_PATH"] = str(cudnn_lib_path) + ":" + os.environ.get("LD_LIBRARY_PATH", "") 
except:
    pass
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')),file=sys.stderr)

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

tf.config.optimizer.set_jit(True)  # Enable XLA.

# List of data files
data_files = ['Data1000new.xlsx']  # Add more files if needed

# Load data from Excel files
for file_path in data_files:
    # Load all sheets from the Excel file
    strain_data, sigma_data, time_data = load_excel_data(file_path)  # Changed to get strain data directly

# Convert data to tensors
strain = tf.convert_to_tensor(strain_data, dtype=tf.float32)
sigma = tf.convert_to_tensor(sigma_data, dtype=tf.float32)
time = tf.convert_to_tensor(time_data, dtype=tf.float32)

# Store dimensions
n_datasets = strain.shape[0]
timesteps = strain.shape[1]
timesteps_val = timesteps

# Calculate invariants from strain tensor
strain_invariants = calculate_invariants(strain)  # [batch, timesteps, invariants]
invariants = strain_invariants.shape[2]

# Calculate derivatives with respect to strain
strain_derivatives = calculate_stress_derivatives(strain)

# Create training data
# Input: strain invariants and timestep
train_x = tf.concat([strain_invariants, tf.expand_dims(time, -1)], axis=-1)

# Output: stress components at t+1 (shift stress tensor one step back)
train_y = sigma


# Call the plotting function with our data
plot_detailed_data(strain, sigma)

# Plot unnormalized data before normalization
print("\nPlotting unnormalized data...")
plot_unnormalized_data(train_x, train_y)
print("Plot saved as 'unnorm_data.png' in the plots directory")

# Plot unnormalized derivatives before normalization
print("\nPlotting unnormalized stress derivatives...")
plot_unnormalized_derivatives(strain_derivatives)
print("Plot saved as 'unnorm_deriv.png' in the plots directory")

# Normalization
print("Normalization",file=sys.stderr) 
normalized_train_x, s_x, m_x = normalize_invariants(train_x)
normalized_strain_derivatives, s_deriv, m_deriv = normalize_stress_derivatives(strain_derivatives)
strain_derivatives = normalized_strain_derivatives  # Replace the original tensor
normalized_train_y, s_y, m_y = normalize_stresses(train_y)

# Plot normalized data right after normalization
print("\nPlotting normalized data...")
plot_normalized_data(normalized_train_x, normalized_train_y)
print("Plot saved as 'norm_data.png' in the plots directory")

# Plot normalized derivatives
print("\nPlotting normalized stress derivatives...")
plot_normalized_derivatives(normalized_strain_derivatives)
print("Plot saved as 'norm_deriv.png' in the plots directory")

# Store scaling factors for later use
s_all = s_x
m_all = m_x
s_out = s_y
m_out = m_y
s_dt = s_x[6]
m_dt = m_x[6]

# Convert normalized data to tensorflow tensors
train_x_tf = tf.convert_to_tensor(normalized_train_x, dtype=tf.float32)
train_y_tf = tf.convert_to_tensor(normalized_train_y, dtype=tf.float32)
train_derivatives_tf = tf.convert_to_tensor(normalized_strain_derivatives, dtype=tf.float32)

# Set batch size
batch_size = 4  # Will give 3 updates per epoch with 12 datasets

# Create dataset that includes all normalized components
train_dataset = tf.data.Dataset.from_tensor_slices((
    train_x_tf,
    train_y_tf,
    train_derivatives_tf
))
# Shuffle with buffer size equal to dataset size for perfect shuffling
train_dataset = train_dataset.shuffle(buffer_size=n_datasets, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(batch_size)

# Define all directories to creates
directories = [
    './stress_exact',
    './final_predictions',
    './input',
    './strain',
]

# Create or recreate each directory
for dir_path in directories:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

print("Output directories created successfully!")
print("Training on experimental data!",file=sys.stderr)

# Hyperparameters
layer_size = 60
layer_size_fenergy = 60 
internal_variables = 12
num_epochs = 30000
 
 
# Learning rate scheduler
initial_learning_rate = 0.01
decay_steps = num_epochs // 10
decay_rate = 0.9

# Create learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True  # Make the decay steps discrete rather than continuous
)

# Create a new optimizer instance with the learning rate schedule
train_op_experimental = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    clipnorm=1.0
)

# Create the model with the full stress derivatives tensor
pred_model = PIDL(s_all, m_all, s_out, m_out, layer_size, internal_variables, 
                 layer_size_fenergy, batch_size, train_op_experimental, 
                 s_dt, m_dt, strain_derivatives, invariants, use_picnn=False, training_silent=True)

# Print initial learning rate
print(f"\nInitial learning rate: {initial_learning_rate}")

# Train the model using the dataset
pred_model.network_learn(train_dataset, num_epochs)

# Save the trained model (configuration and weights) using the custom save method
model_save_path = 'models/trained_model'
pred_model.save(model_save_path)
print(f"\nModel saved to {model_save_path}")

# Save normalization values
normalization_path = os.path.join('models', 'normalization_values.npz')
np.savez(normalization_path,
         s_all=s_all.numpy().astype(np.float32),
         m_all=m_all.numpy().astype(np.float32),
         s_out=s_out.numpy().astype(np.float32),
         m_out=m_out.numpy().astype(np.float32),
         s_dt=s_dt.numpy().astype(np.float32),
         m_dt=m_dt.numpy().astype(np.float32))
print(f"Normalization values saved to {normalization_path}")

# Get predictions
normalized_train_x_tf = tf.convert_to_tensor(normalized_train_x, dtype=tf.float32)
stress_predict, psi, dissipation_rate, internal_vars = pred_model.obtain_output(normalized_train_x_tf)
stress_predict_un = stress_predict * s_y

# Save the trained values
for i in range(train_x.shape[0]):
    filename1 = f"./final_predictions/fnergy_{i}.txt"  # Naming each file as array_0.txt, array_1.txt, ...
    filename2 = f"./final_predictions/zi_{i}.txt"
    filename3 = f"./final_predictions/diss_{i}.txt" 
    filename4 = f"./final_predictions/stress_pred_{i }.txt" 
    filename5 = f"./stress_exact/stress_{i}.txt" 
    filename6 = f"./input/input_{i}.txt" 
    filename7 = f"./strain/strain_{i}.txt" 
    
    # Convert tensors to numpy arrays before reshaping
    psi_np = psi[i].numpy()
    internal_vars_np = internal_vars[i].numpy()
    dissipation_rate_np = dissipation_rate[i].numpy()
    stress_predict_np = stress_predict_un[i].numpy()
    train_y_np = train_y[i].numpy()
    train_x_np = train_x[i].numpy()
    strain_np = strain[i].numpy()
    
    np.savetxt(filename1, psi_np.reshape(-1), delimiter=',', fmt='%f')    
    # Reshape the 2D slice and then flatten it
    np.savetxt(filename2, internal_vars_np.reshape(-1), delimiter=',', fmt='%f')
    np.savetxt(filename3, dissipation_rate_np.reshape(-1), delimiter=',', fmt='%f')
    np.savetxt(filename4, stress_predict_np.reshape(-1), delimiter=',', fmt='%f')
    np.savetxt(filename5, train_y_np.reshape(-1), delimiter=',', fmt='%f')
    np.savetxt(filename6, train_x_np.reshape(-1), delimiter=',', fmt='%f')
    np.savetxt(filename7, strain_np.reshape(-1), delimiter=',', fmt='%f')

print("\n... Output for training data printed out!")


# After training and prediction, call the plotting function
print("\nPlotting predicted vs true stress-strain behavior...")
plot_predicted_vs_true(strain, train_y, stress_predict, s_y, m_y)
print("Plot saved as 'pred_vs_true.png' in the plots directory")

