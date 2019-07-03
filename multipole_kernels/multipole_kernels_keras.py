import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras.layers import Layer


class MultipoleKernel(Layer):
    def __init__(self, kernel_size=[3, 3], ℓ=[0, 1], input_filters=1,
                 output_filters=None, padding="VALID", **kwargs):
        self.kernel_size = kernel_size
        self.dimension = len(kernel_size)
        if self.dimension > 3:
            print("Maximum dimension is 3")
            sys.exit()
        self.ℓ = ℓ
        if output_filters is None:
            self.output_filters = [1 for this_ℓ in ℓ]
        elif len(output_filters) != len(ℓ):
            print("A filter imust provided for each ℓ")
            sys.exit()
        else:
            self.output_filters = output_filters
        self.padding = padding
        super(MultipoleKernel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_input_filters = input_shape[-1]
        self.indices, self.weight_index = self.get_indices()
        self.num_params = self.weight_index[-1].astype(np.int) + 1
        self.num_output_filters = self.indices[-1, -1].astype(np.int) + 1
        self.w = self.add_weight(
            name="weights",
            shape=(self.num_params,),
            initializer="normal",
            trainable=True)
        self.b = self.add_weight(
            name="biases",
            shape=(self.num_output_filters,),
            initializer='constant',
            trainable=True)
        weight_index = tf.constant(
            self.weight_index,
            dtype=tf.int32,
            name="weight_index")
        self.full_weights = tf.gather(self.w,
                                      weight_index,
                                      name="full_weights")
        indices = tf.constant(self.indices,
                              dtype=tf.int32,
                              name="indices")
        shape = tf.constant(self.kernel_size + [self.num_input_filters,
                                                self.num_output_filters],
                            dtype=tf.int32,
                            name="shape")
        self.kernel = tf.scatter_nd(indices,
                                    self.full_weights,
                                    shape, name="kernel")
        self.kernel.set_shape(self.kernel_size + [self.num_input_filters,
                                                  self.num_output_filters])
        super(MultipoleKernel, self).build(input_shape)

    def get_indices(self):
        grid, distance = self.distances()
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.dimension == 1:
                self.θ = grid[0]
            elif self.dimension == 2:
                self.θ = np.arctan(np.abs(grid[1] / grid[0]))
            elif self.dimension == 3:
                self.θ = np.arccos(grid[2] / distance)
                self.ϕ = np.arctan(grid[1] / grid[0])

        indices = np.zeros((0, self.dimension + 2))
        weight_index = np.zeros((0))
        filter_counter = 0
        weight_counter = 0
        for ℓ in range(len(self.ℓ)):
            if self.dimension == 1:
                m_range = range(1)
            elif self.dimension == 2:
                m_range = range(self.ℓ[ℓ] + 1)
            elif self.dimension == 3:
                m_range = range(-self.ℓ[ℓ], self.ℓ[ℓ]+1)
            for m in m_range:
                distance_modifier = self.angular(self.ℓ[ℓ], m)
                distance_modifier[np.isnan(distance_modifier)] = 0.
                this_distance = distance * distance_modifier
                these_unique = np.unique(this_distance)
                if ℓ != 0 and these_unique[0] == 0:
                    these_unique = np.delete(these_unique, 0)
                for dist in these_unique:
                    temp_indices = np.argwhere(dist == this_distance)
                    num_elements = temp_indices.shape[0]
                    for in_filt in range(self.num_input_filters):
                        for out_filt in range(self.output_filters[ℓ]):
                            indices = np.concatenate(
                                (indices,
                                 np.append(temp_indices,
                                           np.tile([in_filt,
                                                    filter_counter + out_filt],
                                                   [num_elements, 1]),
                                           axis=1)))
                            weight_index = np.concatenate(
                                (weight_index,
                                 np.tile(weight_counter, [num_elements])))
                            weight_counter += 1
                filter_counter += self.output_filters[ℓ]
        return indices, weight_index

    def angular(self, ℓ, m):
        if self.dimension == 1:
            if self.θ.shape[0] % 2 == 0:
                return np.concatenate(
                    [-np.ones(self.θ.shape[0]//2),
                     np.ones(self.θ.shape[0]//2)])**ℓ
            else:
                return np.concatenate(
                    [-np.ones(self.θ.shape[0]//2),
                     [0],
                     np.ones(self.θ.shape[0]//2)])**ℓ
        elif self.dimension == 2:
            return np.cos(ℓ * self.θ)
        elif self.dimension == 3:
            return sph_harm(m, ℓ, self.θ, self.ϕ).real

    def distances(self):
        full_distance = np.mgrid[
            tuple(
                slice(-self.kernel_size[dim] / 2 + 0.5,
                      self.kernel_size[dim] / 2 + 0.5,
                      1)
                for dim in range(self.dimension))]
        return full_distance, np.sqrt(np.sum(full_distance**2., 0))

    def call(self, x):
        return tf.add(tf.nn.convolution(x,
                                        self.kernel,
                                        self.padding),
                      self.b,
                      name="multipole_convolution")

    def compute_output_shape(self, input_shape):
        current_shape = input_shape[1:-1]
        if self.padding == "VALID":
            for dim in range(len(current_shape)):
                current_shape[dim] += 1 - self.kernel_size[dim]
        return tuple([
                [input_shape[0]] + current_shape + [self.num_output_kernels]])
