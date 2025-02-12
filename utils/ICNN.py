import tensorflow as tf
from tensorflow import keras
import numpy as np

class FICNN(tf.keras.Model):
    def __init__(self, layer_sizes, activation='softplus', dtype=tf.float64, name='ficnn'):
        """
        Fully Input Convex Neural Network
        Args:
            layer_sizes: list of integers, the size of each hidden layer
            activation: activation function (must be convex and non-decreasing)
            dtype: data type for the layers
            name: name of the model
        """
        super(FICNN, self).__init__(name=name)
        
        self.n_layers = len(layer_sizes)
        self.activation = getattr(tf.nn, activation)
        
        # Initialize layers
        self.z_layers = []  # W^(z) weights for hidden states
        self.y_layers = []  # W^(y) weights for input passthrough
        self.biases = []    # bias terms
        
        # First layer is special (no z input)
        self.y_layers.append(tf.keras.layers.Dense(
            layer_sizes[0],
            use_bias=True,
            dtype=dtype,
            name=f'y_layer_0'
        ))
        
        # Remaining layers
        for i in range(1, self.n_layers):
            # W^(z) must be non-negative
            self.z_layers.append(tf.keras.layers.Dense(
                layer_sizes[i],
                use_bias=False,
                kernel_constraint=tf.keras.constraints.NonNeg(),
                dtype=dtype,
                name=f'z_layer_{i}'
            ))
            
            # W^(y) can be any real value
            self.y_layers.append(tf.keras.layers.Dense(
                layer_sizes[i],
                use_bias=True,
                dtype=dtype,
                name=f'y_layer_{i}'
            ))
    
    def call(self, inputs):
        """
        Forward pass of the FICNN
        Args:
            inputs: input tensor y
        Returns:
            output of the network
        """
        y = inputs
        z = self.y_layers[0](y)
        z = self.activation(z)
        
        # Remaining layers with both z and y connections
        for i in range(1, self.n_layers):
            z_new = self.z_layers[i-1](z) + self.y_layers[i](y)
            if i < self.n_layers - 1:
                z = self.activation(z_new)
            else:
                z = z_new  # No activation on the last layer
        
        return z
