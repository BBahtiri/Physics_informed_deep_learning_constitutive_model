import tensorflow as tf
from tensorflow import keras
import numpy as np

class PICNN(tf.keras.Model):
    def __init__(self, x_sizes, y_sizes, internal_variables, invariants, activation='softplus', x_activation='softplus', name='picnn', dtype=tf.float64):
        """
        Partially Input Convex Neural Network (PICNN)
        
        This implementation follows the mathematical formulation where:
        - Non-convex path (v_l): Regular feedforward path
        - Convex path (u_l): Path that ensures convexity in specified inputs
        
        Convexity conditions:
        1. All weights in W^uu must be non-negative
        2. All activations A^U must be convex
        3. All activations A^U must be non-decreasing
        4. All activations A^uv must map to non-negative values
        
        Args:
            x_sizes: list of integers, sizes of hidden layers in non-convex path (v_l)
            y_sizes: list of integers, sizes of hidden layers in convex path (u_l)
            activation: activation for convex path (must be convex and non-decreasing)
            x_activation: activation for non-convex path
            name: name of the model
        """
        super(PICNN, self).__init__(name=name)
        
        assert len(x_sizes) == len(y_sizes), "Both paths must have same number of layers"
        self.n_layers = len(x_sizes)
        self.activation = getattr(tf.nn, activation)
        self.x_activation = getattr(tf.nn, x_activation)
        self.y_activation = getattr(tf.nn, activation)
        self._dtype = dtype
        
        # Initialize layers for non-convex path (v_l)
        self.W_vv = []  # Weights for non-convex path
        
        # Initialize layers for convex path (u_l)
        self.W_uu = []    # Weights for convex path (must be non-negative)
        self.W_uiu = []   # Weights for convex input interaction (l≥2)  
        self.W_uv = []   # Weights for convex input interaction (l≥2)
        self.W_vvtil = []  # Weights for input-path interaction in later layers
        self.W_iuvtil = []  # Weights for input-path interaction in first layer
        
        # Build layers
        for i in range(self.n_layers):
            # Non-convex path layers (v_l)
            self.W_vv.append(tf.keras.layers.Dense(
                x_sizes[i],
                use_bias=True,
                dtype=self._dtype,
                name=f'W_vv_{i}'
            ))
            
            # Convex path layers (u_l)
            # W_uu must be non-negative for convexity
            self.W_uu.append(tf.keras.layers.Dense(
                y_sizes[i],
                use_bias=True,
                kernel_constraint=tf.keras.constraints.NonNeg(),
                dtype=self._dtype,
                name=f'W_uu_{i}'
            ))
            
            # Interaction weights
            self.W_uiu.append(tf.keras.layers.Dense(
                y_sizes[i],
                use_bias=True,
                dtype=self._dtype,
                name=f'W_uv_{i}'
            ))
            
            # First layer input interaction
            self.W_uv.append(tf.keras.layers.Dense(
                y_sizes[i],
                use_bias=True,
                dtype=self._dtype,
                name=f'W_iuv_{i}'
            ))

            self.W_vvtil.append(tf.keras.layers.Dense(
                y_sizes[i],
                use_bias=True,
                dtype=self._dtype,
                name=f'W_vvtil_{i}'
            ))

            self.W_iuvtil.append(tf.keras.layers.Dense(
                y_sizes[i],
                use_bias=True,
                dtype=self._dtype,
                name=f'W_iuvtil_{i}'
            ))


            

    
    def call(self, inputs):
        """
        Forward pass following the PICNN formulation:
        
        Non-convex path:
            v_1 = A_1^-(W_1^vv i^- + b_1^vv)
            v_l = A_l^-(W_l^vv v_{l-1} + b_l^vv)
            
        Convex path:
            u_1 = A_1^U(W_1^uu[i^U ⊙ A_1^{i^Uv}(W_1^{i^Uv}i^- + b_1^{i^Uv})] + W_1^uv i^- + b_1^u)
            u_l = A_l^U(W_l^uu[u_{l-1} ⊙ A_l^uv(W_l^uv v_{l-1} + b_l^uv)] + 
                       W_l^ui^U[i^U ⊙ A_l^{i^Uv}(W_l^{i^Uv}v_{l-1} + b_l^{i^Uv})] + 
                       W_l^uv v_{l-1} + b_l^u)
        
        Args:
            inputs: tuple of (i^-, i^U) where i^- are non-convex inputs and i^U are convex inputs
        Returns:
            u_L: output of the convex path
        """
        i_minus, i_U = inputs  # non-convex and convex inputs
        
        # Cast inputs
        v = tf.cast(i_minus, self._dtype)  # non-convex path
        u = tf.cast(i_U, self._dtype)      # convex path
        
        # Forward pass
        for i in range(self.n_layers):
            # Non-convex path update
            v_next = self.W_vv[i](v)
            v = self.x_activation(v_next)
            
            if i == 0:
                # First layer (l=1)
                # Compute i^U ⊙ A_1^{i^Uv}(W_1^{i^Uv}i^- + b_1^{i^Uv})
                iuv_term = self.W_iuvtil[i](i_minus)
                iuv_act = self.y_activation(iuv_term)
                u_term = i_U * iuv_act
                u_term_1 = self.W_uiu[i](u_term)
                u_term_2 = self.W_uv[i](i_minus)

                # Apply non-negative weights and combine
                u = self.activation(u_term_1 + u_term_2)
            else:
                # Later layers (l≥2)
                # First term: u_{l-1} ⊙ A_l^uv(W_l^uv v_{l-1} + b_l^uv)
                uv_term = self.W_vvtil[i](v)
                uv_act = self.y_activation(uv_term)
                u_term = u * uv_act
                u_term_1 = self.W_uu[i](u_term)
                
                # Second term: i^U ⊙ A_l^{i^Uv}(W_l^{i^Uv}v_{l-1} + b_l^{i^Uv})
                iuv_term = self.W_iuvtil[i-1](v)
                iuv_act = self.y_activation(iuv_term)
                iu_term = i_U * iuv_act
                iu_term_1 = self.W_uiu[i](iu_term)
                iu_term_2 = self.W_uv[i](v)

                # Combine all terms
                u = self.activation(u_term_1 + iu_term_1 + iu_term_2)
        
        return u  # Return final layer of convex path
