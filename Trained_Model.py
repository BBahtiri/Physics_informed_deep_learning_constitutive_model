#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script loads the trained PIDL model weights from models/trained_model and
loads the normalization values from models/normalization_values.npz.
It then loads the training data (from an Excel file), applies the saved normalization,
runs model inference, and finally plots the predicted vs true stress–strain behavior.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tensorflow as tf
import numpy as np

from utils.PIDL import PIDL
from utils.misc import load_excel_data, calculate_invariants, calculate_stress_derivatives, plot_predicted_vs_true

def main():
    # ---------------------------
    # 1. Load Normalization Values
    # ---------------------------
    norm_file = 'models/normalization_values.npz'
    norm_data = np.load(norm_file)
    # These keys were saved in Main_ML.py
    s_all = norm_data['s_all']  # scaling factors for invariants & dt, shape (7,)
    m_all = norm_data['m_all']  # min values for invariants & dt, shape (7,)
    s_out = norm_data['s_out']  # scaling factors for stresses, shape (6,)
    m_out = norm_data['m_out']  # mean values for stresses, shape (6,)
    s_dt = norm_data['s_dt']    # scaling for dt (redundant if dt is kept unnormalized)
    m_dt = norm_data['m_dt']    # mean for dt

    print("Normalization values loaded:")
    print("s_all:", s_all)
    print("m_all:", m_all)
    print("s_out:", s_out)
    print("m_out:", m_out)

    # ---------------------------
    # 2. Load the Trained Model
    # ---------------------------
    model_path = 'models/trained_model'
    trained_model = PIDL.load(model_path)
    print("Trained model loaded successfully.")

    # ---------------------------
    # 3. Load Training Data from Excel
    # ---------------------------
    # For example, we use a file with your training data:
    data_files = ['Data1000new.xlsx']
    # (Assuming one Excel file with multiple sheets)
    for file_path in data_files:
        strain_data, sigma_data, time_data = load_excel_data(file_path)
    # Convert data arrays to tensors
    strain = tf.convert_to_tensor(strain_data, dtype=tf.float32)  # shape [batch, timesteps, 6]
    sigma = tf.convert_to_tensor(sigma_data, dtype=tf.float32)    # shape [batch, timesteps, 6]
    time_tensor = tf.convert_to_tensor(time_data, dtype=tf.float32)  # shape [batch, timesteps]

    # ---------------------------
    # 4. Build the Model Input
    # ---------------------------
    # Calculate invariants from strain (using the provided helper)
    strain_invariants = calculate_invariants(strain)  # shape [batch, timesteps, 6]
    # Concatenate the invariants and the timestep to form input: [invariants, dt]
    # This makes a tensor of shape [batch, timesteps, 7]
    train_x = tf.concat([strain_invariants, tf.expand_dims(time_tensor, -1)], axis=-1)

    # ---------------------------
    # 5. Normalize the Input Using Saved Values
    # ---------------------------
    # Note: In training, invariants were normalized (min–max to [-1,1]) while dt was left unchanged.
    s_all_tf = tf.convert_to_tensor(s_all, dtype=tf.float32)  # shape (7,)
    m_all_tf = tf.convert_to_tensor(m_all, dtype=tf.float32)  # shape (7,)
    # Normalize only the first 6 components (the invariants)
    normalized_invariants = (train_x[..., :6] - m_all_tf[:6]) / s_all_tf[:6] * 2 - 1
    # The timestep (last column) remains unchanged
    dt_component = train_x[..., 6:]
    normalized_train_x = tf.concat([normalized_invariants, dt_component], axis=-1)

    # ---------------------------
    # NEW: Recalculate Stress Derivatives
    # ---------------------------
    # Recalculate stress derivatives using the original strain data
    stress_derivatives = calculate_stress_derivatives(strain_data)

    # ---------------------------
    # 6. Run the Trained Model on the Training Data
    # ---------------------------
    # Obtain the model outputs:
    # f_σ_tdt: predicted stress (normalized),
    # psi: free energy,
    # diss_rate: dissipation rate, and
    # internal_vars: predicted internal variables.
    stress_predict_norm, psi, diss_rate, internal_vars = trained_model.obtain_output(normalized_train_x, stress_derivatives)
    
    # ---------------------------
    # 7. Unnormalize the Predicted Stress
    # ---------------------------
    # The original training stress tensor (sigma) was normalized using s_out.
    # We unnormalize the predictions by multiplying by s_out.
    s_out_tf = tf.convert_to_tensor(s_out, dtype=tf.float32)
    stress_predict = stress_predict_norm * s_out_tf[None, None, :]
    
    # ---------------------------
    # 8. Evaluate and Plot Results
    # ---------------------------
    mse = tf.reduce_mean(tf.square(stress_predict - sigma))
    print("Mean Squared Error (MSE) on training data:", mse.numpy())

    # Plot true vs predicted stress–strain behavior.
    # The plotting function expects the strain, true stress, normalized predicted stress,
    # and the stress normalization factors (s_out, m_out) so that it can unnormalize internally.
    plot_predicted_vs_true(strain, sigma, stress_predict_norm, s_out, m_out)
    print("Predicted vs True stress–strain plot saved (check the plots directory).")

if __name__ == '__main__':
    main()
