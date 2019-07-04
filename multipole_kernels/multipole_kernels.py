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
"""


__version__ = '0.1a'
__author__ = "Tom Charnock"

import numpy as np
from scipy.special import sph_harm
import tensorflow as tf
import sys


class multipole_kernels():
    """Multipole Kernels

    Attributes:
    kernel_size -- list -- nd shape of kernel
    dimension -- int -- number of dimensions for the kernel (max = 3)
    ℓ -- list -- multipole values to create kernels
    num_input_filters -- int -- number of input filters
    output_filters -- list -- number of output filters/multipole
    num_params -- int -- number of independent weights for kernel
    num_output_filters -- int -- total number of output filters (for biases)
    w -- tensor -- set of independent variables/placeholder for kernel
    b -- tensor -- set of independent variables/placeholder for biases
    kernel -- tensor -- multipole kernels with filters as different multipoles
    indices -- ndarray (num_indices, dimension+2) -- weight kernel indices
    weight_index -- nd_array (num_indices,) -- indices to gather weights at
    """
    def __init__(self, kernel_size=[3, 3], ℓ=[0, 1], input_filters=1,
                 output_filters=None, placeholder=None, keras=None):
        """ Builds kernels with weights shared according to symmetries

        Calls to:
        get_indices() -- gets indices to place same weights at in kernel
        get_weights() -- gets independent weights to be scattered into kernel
        build_kernel() -- puts weights at the correct indices in kernel tensor

        Keyword arguments:
        kernel_size -- list -- nd shape of kernel (default [3, 3])
        ℓ -- list -- multipole values to create kernels (default [0, 1])
        input_filters -- int -- number of input filters (default 1)
        output_filters -- list -- number of output filters per multipole
        placeholder -- str -- kernel tensor name, Variable used if not provided
        keras -- list -- if using keras this initialises the weights and biases

        Parameters:
        kernel_size -- list -- nd shape of kernel
        dimension -- int -- number of dimensions for the kernel (max = 3)
        ℓ -- list -- multipole values to create kernels
        num_input_filters -- int -- number of input filters
        output_filters -- list -- number of output filters/multipole
        indices -- ndarray (num_indices, dimension+2) -- weight kernel indices
        weight_index -- nd_array (num_indices,) -- indices to gather weights at
        num_params -- int -- number of independent weights for kernel
        num_output_filters -- int -- total number of output filters for biases
        w -- tensor -- independent variables/placeholder for kernel
        b -- tensor -- independent variables/placeholder for biases
        indices_t -- tensor -- weight kernel indices as a tensor
        weight_index_t -- tensor -- indices to gather weights as a tensor
        shape -- tensor -- shape of the convolutional kernel
        kernel -- tensor -- kernels with filters as different multipoles
        """
        self.kernel_size = kernel_size
        self.dimension = len(kernel_size)
        if self.dimension > 3:
            print("Maximum dimension is 3")
            sys.exit()
        self.ℓ = ℓ
        self.num_input_filters = input_filters
        if output_filters is None:
            self.output_filters = [1 for this_ℓ in ℓ]
        elif len(output_filters) != len(ℓ):
            print("A filter must provided for each ℓ")
            sys.exit()
        else:
            self.output_filters = output_filters
        indices, weight_index = self.get_indices()
        self.num_params = weight_index[-1].astype(np.int) + 1
        self.num_output_filters = indices[-1, -1].astype(np.int) + 1
        if not keras:
            self.w, self.b, indices_t, weight_index_t, shape = \
                self.get_weights(indices, weight_index, placeholder)
            self.kernel = self.build_kernel(indices_t,
                                            weight_index_t,
                                            shape,
                                            self.w)
        else:
            self.indices = indices
            self.weight_index = weight_index

    def get_weights(self, indices, weight_index, placeholder):
        """ Gets TensorFlow variables (placeholders) for weights and indices

        We create a TensorFlow variable (or placeholder) with as many
        parameters as there are independent weights in the kernels. We also
        create the tensor forms of the indices and weight indices

        Called by:
        __init__() -- finds indices and places weights in kernel

        Arguments:
        placeholder -- str -- kernel tensor name, Variable used if not provided

        Returns:
        w -- tensor -- independent variables/placeholder for kernel
        b -- tensor -- independent variables/placeholder for biases
        indices_t -- tensor -- weight kernel indices as a tensor
        weight_index_t -- tensor -- indices to gather weights as a tensor
        shape - tensor - shape of the convolutional kernel
        """
        if placeholder is None:
            w = tf.Variable(np.random.normal(0, 1, self.num_params),
                            dtype=tf.float32,
                            name="weights")
            b = tf.Variable(np.zeros(self.num_output_filters),
                            dtype=tf.float32,
                            name="biases")
        else:
            w = tf.placeholder(dtype=tf.float32,
                               shape=(self.num_params),
                               name=placeholder+"/weights")
            b = tf.placeholder(dtype=tf.float32,
                               shape=(self.num_output_filters),
                               name=placeholder+"/biases")
        indices_t = tf.Variable(indices,
                                dtype=tf.int32,
                                trainable=False)
        weight_index_t = tf.Variable(weight_index,
                                     dtype=tf.int32,
                                     trainable=False)
        shape = tf.Variable(self.kernel_size + [self.num_input_filters,
                                                self.num_output_filters],
                            dtype=tf.int32,
                            trainable=False)
        return w, b, indices_t, weight_index_t, shape

    def build_kernel(self, indices, weight_index, shape, w):
        """ Puts same weights at the correct indices in kernel tensor

        Using the indices which we calculated in get_indices() and the weights
        from get_weights() we can construct the radial (multipole expansion)
        kernels. These weights are first combined into a tensor whoes dimension
        is the same as the number of indices. These are then scattered into the
        convolutional kernel. Here we also get the biases kernels, one
        parameter for each output filter.

        Called by:
        __init__() -- finds indices and places weights in kernel

        Arguments:
        indices -- tensor -- kernel index postions
        weight_index -- tensor -- indices to gather weights
        shape - tensor -- shape of the convolutional kernel
        w -- tensor -- independent variables/placeholder for kernel

        Parameters:
        full_weights -- tensor -- weights gathered at correct indices

        Returns:
        kernel -- tensor -- multipole kernels, filters as different multipoles
        """
        full_weights = tf.gather(w, weight_index)
        kernel = tf.scatter_nd(indices, full_weights, shape)
        kernel.set_shape(self.kernel_size + [self.num_input_filters,
                                             self.num_output_filters])
        return kernel

    def get_indices(self):
        """ Gets indices for equidistant/independent points in kernel

        We want to get the indices of a convolutional kernel at every
        equidistant point in the kernel (ℓ = 0), or every independent point
        (ℓ > 1). To do this we first find the distance from the centre of the
        grid and make the coordinate transformation into spherical polar
        coordinates. Using these gridded distances (angles) we find the indices
        of each unique distance (ℓ = 0) or independent point after calculating
        either the Fourier series (2D) or spherical harmonic coeffcients (3D)
        (ℓ > 0) o the grid. These indices are collected for each multipole
        moment and every input and output filter and appended to the array.
        We also collect the which indices are associated with which independent
        weight.

        Called by:
        __init__() -- finds indices and places weights in kernel

        Calls to:
        get_distance() -- calculates distances from centre of kernel on grid
        get_angles() -- distances on grid to angles from the centre of kernel
        get_symmetries() -- calculates nd symmetries to place like weights at

        Returns:
        indices -- ndarray (num_indices, dimension+2) -- indices in kernel
        weight_index -- ndarray (num_indices,) -- indices to gather weights

        Parameters:
        grid -- ndarray ([dimension] + kernel_size) -- nd distance from centre
        distance -- ndarray (kernel_size) -- Euclidean distance from centre
        angles -- dict -- angles from centre of kernel evaluated on kernel grid
        indices -- ndarray (num_indices, dimension+2) -- indices in kernel
        weight_index -- ndarray (num_indices,) -- indices to gather weights at
        filter_counter -- int -- counter for the total number of filters
        weight_counter -- int -- counter for the total number of weights
        ℓ -- int -- loop variable for each multipole moment
        m_range -- range -- a range object different for counting nd m values
        m -- int -- loop variable for each m when counting multipoles
        distance_modifier -- ndarray (kernel_size) -- symmetries on kernel grid
        this_distance -- ndarray (kernel_size) -- modified distance from centre
        these_unique -- ndarray (num_elements,) -- unique elements in kernel
        dist -- float -- loop variable, each unique modified distance in kernel
        temp_indices -- ndarray (num_elements, dimension) -- indices of unique
        num_elements -- int -- number of unique elements in the modified grid
        in_filt -- int -- loop variable counting the number of input filter
        out_filter -- int -- loop variable counting output filter per multipole
        """
        grid, distance = self.get_distance()
        angles = self.get_angles(grid, distance)
        indices = np.zeros((0, self.dimension + 2))
        weight_index = np.zeros((0))
        filter_counter = 0
        weight_counter = 0
        for ℓ in range(len(self.ℓ)):
            if self.dimension == 1:
                # There is only one rotation in 1D (a sign flip)
                m_range = range(1)
            elif self.dimension == 2:
                # There are nd rotations in 2D
                m_range = range(self.ℓ[ℓ] + 1)
            elif self.dimension == 3:
                # There are 2ℓ + 1 elements in 3D
                m_range = range(-self.ℓ[ℓ], self.ℓ[ℓ]+1)
            for m in m_range:
                if self.ℓ[ℓ] > 0:
                    distance_modifier = self.get_symmetries(self.ℓ[ℓ],
                                                            m, angles)
                    distance_modifier[np.isnan(distance_modifier)] = 0.
                    this_distance = distance * distance_modifier
                else:
                    this_distance = distance
                these_unique = np.unique(this_distance)
                if self.ℓ[ℓ] != 0 and these_unique[0] == 0:
                    # We do not need weights at the where the grid is zero.
                    these_unique = np.delete(these_unique, 0)
                for dist in these_unique:
                    temp_indices = np.argwhere(dist == this_distance)
                    num_elements = temp_indices.shape[0]
                    for in_filt in range(self.num_input_filters):
                        for out_filt in range(self.output_filters[ℓ]):
                            # The input and output filter indices are appended
                            # to the kernel position index to put weight in
                            # correct filter
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

    def get_angles(self, grid, distance):
        """ Calculate angles from centre of the kernel in nd.

        In 2D and 3D we transform the Cartesian coordinate system to
        (spherical) polar coordinates. In 1D we just return the 1D grid.

        Called by:
        get_indices() -- gets indices for equidistant/independent kernel points

        Inputs:
        grid -- ndarray [dimension] + kernel_size -- nd distance from centre
        distance -- ndarray kernel_size -- Euclidean distance from centre

        Returns:
        dict -- (spherical) polar coordinates on the grid
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.dimension == 1:
                return {"x": grid[0]}
            elif self.dimension == 2:
                return {"θ": np.arctan(np.abs(grid[1] / grid[0]))}
            elif self.dimension == 3:
                return {"θ": np.arccos(grid[2] / distance),
                        "ϕ": np.arctan(grid[1] / grid[0])}

    def get_symmetries(self, ℓ, m, angles):
        """ Calculate the radial symmetries which can distort the kernel

        In 1D, any ℓ = 0 is symmetric about the centre and needs no
        modification. Any ℓ > 0 recovers the non-symmetric kernel and so we can
        just modify the distances by a range. In 2D we can use the Fourier
        basis to modify the distances which breaks the radial symmetry
        according to the value of ℓ. ℓ = 0 is radial since the cofficient is 1,
        but any ℓ > 0 induces a rotational phase which modifies the distance.
        In 3D we can use spherical harmonics to represent the same thing.

        Called by:
        get_indices() -- gets indices for equidistant/independent kernel points

        Arguments:
        ℓ -- int -- nd multipole moment
        m -- int -- element (rotation) of the coefficient
        angles -- dict -- angles from centre of kernel evaluated on kernel grid

        Returns:
        ndarray (kernel_size) -- modification symmetry coefficients on a grid
        """
        if self.dimension == 1:
            return np.arange(self.kernel_size[0])
        elif self.dimension == 2:
            return np.exp(np.complex(0., 1.) * angles["θ"] * ℓ).real
        elif self.dimension == 3:
            return sph_harm(m, ℓ, angles["θ"], angles["ϕ"]).real

    def get_distance(self):
        """ Calculates the distance from the centre of the convolutional kernel

        Evaluate the distance of all elements of kernel from the centre of the
        kernel. We will use this to find all elements at the same distance for
        building the kernel with shared weights.

        Called by:
        get_indices() -- gets indices to place same weights at in kernel

        Returns:
        ndarray -- the nd gridded distance from the centre of the kernel
        ndarray -- the gridded Euclidean distance from the centre of the kernel

        Parameters:
        distance -- ndarray ([dimension] + kernel_size) -- gridded nd distance
        """
        distance = np.mgrid[
            tuple(
                slice(-self.kernel_size[dim] / 2 + 0.5,
                      self.kernel_size[dim] / 2 + 0.5,
                      1)
                for dim in range(self.dimension))]
        return distance, np.sqrt(np.sum(distance**2., 0))
