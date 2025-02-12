from .PIDL import PIDL

def calculate_stress_derivatives_trained(strain):
    """Calculate derivatives of invariants with respect to C components."""
    batch_size, timesteps, _ = strain.shape
    derivatives = np.zeros((batch_size, timesteps, 6, 6), dtype=np.float32)
    
    # Define a₀ = [1,0,0]
    a0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # Identity matrix
    I_mat = np.eye(3, dtype=np.float32)
    
    for b in range(batch_size):
        for t in range(timesteps):
            # Construct C matrix from the provided strain (which represents C in this context)
            C_mat = np.zeros((3,3), dtype=np.float32)
            C_mat[0,0] = strain[b,t,0]  # C11
            C_mat[0,1] = C_mat[1,0] = strain[b,t,1]  # C12
            C_mat[0,2] = C_mat[2,0] = strain[b,t,2]  # C13
            C_mat[1,1] = strain[b,t,3]  # C22
            C_mat[1,2] = C_mat[2,1] = strain[b,t,4]  # C23
            C_mat[2,2] = strain[b,t,5]  # C33

            # Calculate needed quantities using float32
            I1 = np.float32(np.trace(C_mat))
            I3 = np.float32(np.linalg.det(C_mat))
            C_inv = np.float32(np.linalg.inv(C_mat))
            Ca0 = np.matmul(C_mat, a0)
            
            # Calculate derivatives
            dI1_dC = I_mat
            dI2_dC = I1 * I_mat - C_mat
            dI3_dC = I3 * C_inv
            dI4_dC = np.outer(a0, a0)
            dI5_dC = np.outer(a0, Ca0) + np.outer(np.matmul(a0, C_mat), a0)
            J = np.float32(np.sqrt(I3))
            dI6_dC = -J * C_inv
            
            # Store derivatives in Voigt notation (order: [11, 12, 13, 22, 23, 33])
            derivatives[b,t,0] = [dI1_dC[0,0], dI1_dC[0,1], dI1_dC[0,2],
                                   dI1_dC[1,1], dI1_dC[1,2], dI1_dC[2,2]]
            derivatives[b,t,1] = [dI2_dC[0,0], dI2_dC[0,1], dI2_dC[0,2],
                                   dI2_dC[1,1], dI2_dC[1,2], dI2_dC[2,2]]
            derivatives[b,t,2] = [dI3_dC[0,0], dI3_dC[0,1], dI3_dC[0,2],
                                   dI3_dC[1,1], dI3_dC[1,2], dI3_dC[2,2]]
            derivatives[b,t,3] = [dI4_dC[0,0], dI4_dC[0,1], dI4_dC[0,2],
                                   dI4_dC[1,1], dI4_dC[1,2], dI4_dC[2,2]]
            derivatives[b,t,4] = [dI5_dC[0,0], dI5_dC[0,1], dI5_dC[0,2],
                                   dI5_dC[1,1], dI5_dC[1,2], dI5_dC[2,2]]
            derivatives[b,t,5] = [dI6_dC[0,0], dI6_dC[0,1], dI6_dC[0,2],
                                   dI6_dC[1,1], dI6_dC[1,2], dI6_dC[2,2]]
    
    return tf.convert_to_tensor(derivatives, dtype=tf.float32) 