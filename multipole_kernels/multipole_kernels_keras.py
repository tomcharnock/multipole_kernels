"""Multipole Kernels

The idea of this module is to reduce (massively) the number of weights
necessary in convolutional kernels for machine learning. The heart of the
procedure is to expand the kernels out in terms of rotational symmetries
which can in principle be truncated at some lower order than a fully
generic kernel.

For example, if all of the information about the data was locally
rotationally invariant then all elements of the convolutional kernel which
are at the same distance from the centre of the kernel would normally
optimise to the same value. In 2D with a 3x3 kernel, one would normally
require 9 weights, but by setting all equidistant elements of the kernel to
the same weight, this reduces to just 3 (centre, faces and edges). In 3D
this reduction is even more massive, a 3x3x3 kernel can be reduced from 27
weights down to just 4 (centre, faces, edges and corners). For more complex
data we can then expand out in multipoles using a Fourier basis (in 2D) or
spherical harmonics (in 3D). This can be expanded arbirarily high, but
should be truncated at most to the degrees of freedom of the full kernel.

One interesting outcome of this expansion (to high orders), is that the
scale of the kernel from input to output of a neural network can be
followed through the multipole pathways to see on which degree of expansion
is most informative about the data.

This module inherits from the true multipole_kernels class and builds a keras
custom layer which can be imported into sequential models, etc.
"""


__version__ = '0.2a'
__author__ = "Tom Charnock"

import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras.layers import Layer
from multipole_kernels.multipole_kernels import multipole_kernels


class MultipoleKernel(Layer):
    """Multipole Kernels

    Attributes:
    kernel_size -- list -- nd shape of kernel
    dimension -- int -- number of dimensions for the kernel (max = 3)
    ℓ -- list -- multipole values to create kernels
    num_input_filters -- int -- number of input filters
    output_filters -- list -- number of output filters/multipole
    num_params -- int -- number of independent weights for kernel
    num_output_filters -- int -- total number of output filters (for biases)
    indices -- ndarray (num_indices, dimension+2) -- weight kernel indices
    weight_index -- nd_array (num_indices,) -- indices to gather weights at
    w -- tensor -- set of independent variables/placeholder for kernel
    b -- tensor -- set of independent variables/placeholder for biases
    kernel -- tensor -- multipole kernels with filters as different multipoles
    build_kernel -- func -- function to construct kernel from multipole_kernels
    padding -- str -- type of padding to use for the convolution
    """
    def __init__(self, kernel_size=[3, 3], ℓ=[0, 1], input_filters=1,
                 output_filters=None, padding="VALID", strides=None, **kwargs):
        """ Initialises multipole_kernels and inherits all of its attributes

        Keyword arguments:
        kernel_size -- list -- nd shape of kernel (default [3, 3])
        ℓ -- list -- multipole values to create kernels (default [0, 1])
        input_filters -- int -- number of input filters (default 1)
        output_filters -- list -- number of output filters per multipole
        padding -- str -- type of padding to use for the convolution

        Parameters:
        mk -- class -- multipole_kernels class for constructing radial kernels
        v -- str -- loop variable containing name of attributes from mk
        """
        mk = multipole_kernels(kernel_size=kernel_size,
                               ℓ=ℓ,
                               input_filters=input_filters,
                               output_filters=output_filters,
                               keras=True)
        for v in vars(mk).keys():
            setattr(MultipoleKernel, v, vars(mk)[v])
        self.build_kernel = mk.build_kernel
        self.padding = padding
        if strides is None:
            self.strides = [1 for i in range(len(kernel_size)+2)]
        else:
            self.strides = strides
        super(MultipoleKernel, self).__init__(**kwargs)

    def build(self, input_shape):
        """ Make keras variables of weights and TensorFlow constants of indices

        Arguments:
        input_shape -- tuple -- the shape of the input to pass to the class

        Parameters:
        w -- tensor -- independent variables/placeholder for kernel
        b -- tensor -- independent variables/placeholder for biases
        indices -- tensor -- weight kernel indices as a tensor
        weight_index -- tensor -- indices to gather weights as a tensor
        shape - tensor - shape of the convolutional kernel
        """
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
        super(MultipoleKernel, self).build(input_shape)
        
    def get_kernel(self):
        """ Returns the current state of the weights scattered into the kernel
        """
        return self.build_kernel(self.indices,
                                 self.weight_index,
                                 self.shape,
                                 self.w)

    def call(self, x):
        """ Performs a convolution using the kernel constructed by the class

        Using the indices calculated in multipole_kernels() we can get keras
        weights from which we can construct the radial (multipole expansion)
        kernels. These weights are first combined into a tensor whoes dimension
        is the same as the number of indices. These are then scattered into the
        convolutional kernel. Here we also get the biases kernels, one
        parameter for each output filter.

        Arguments:
        x -- tensor -- the input tensor to be convolved
        
        Parameters:
        kernel -- tensor -- multipole kernels

        Returns
        tensor -- the convolved input with the radial kernel (plus biases)
        """
        kernel = self.build_kernel(self.indices,
                                   self.weight_index,
                                   self.shape,
                                   self.w)
        return tf.add(tf.nn.convolution(x,
                                        kernel,
                                        strides=self.strides,
                                        padding=self.padding),
                      self.b,
                      name="multipole_convolution")

    def compute_output_shape(self, input_shape):
        """ Computes the output shape of the convolved tensor

        If valid padding is used then we need to subtract the lost edges of the
        tensor, otherwise the tensor stays the same shape.

        Arguments:
        input_shape -- tuple -- the shape of the input tensor

        Parameters:
        current_shape -- tuple -- the shape of the input tensor without filters
        dim -- int -- the counter for the dimension of the input tensor

        Returns:
        tuple -- the shape of the tensor after the convolution
        """
        current_shape = input_shape[:-1]
        if self.padding == "VALID":
            for dim in range(len(current_shape)):
                current_shape[dim] += 1 - self.kernel_size[dim]
        return tuple([current_shape + [self.num_output_kernels]])
