Matrix Manipulation {#matrixmanipulation}
===================

Many different kinds of [matrix manipulation routines](\ref manip_mat) are available:
* tile() to repeat a matrix along dimensions
* join() to concatenate two matrices along a dimension
* [array()](\ref af::array) to adjust the dimensions of an array
* [transpose](\ref af::array::T) a matrix or vector

tile() allows you to repeat a matrix along specified
dimensions, effectively 'tiling' the matrix.  Please note that the
dimensions passed in indicate the number of times to replicate the
matrix in each dimension, not the final dimensions of the matrix.

\snippet test/matrix_manipulation.cpp ex_matrix_manipulation_tile

join() allows you to joining two matrices together.  Matrix
dimensions must match along every dimension except the dimension
of joining (dimensions are 0-indexed). For example, a 2x3 matrix
can be joined with a 2x4 matrix along dimension 1, but not along
dimension 0 since {3,4} don`t match up.

\snippet test/matrix_manipulation.cpp ex_matrix_manipulation_join

Construct a regular mesh grid from vectors `x` and `y`. For example, a
mesh grid of the vectors {1,2,3,4} and {5,6} would result in two matrices:

\snippet test/matrix_manipulation.cpp ex_matrix_manipulation_mesh

[array()](\ref af::array) can be used to create a (shallow) copy of a matrix
with different dimensions.  The number of elements must remain the same as
the original array.

\snippet test/matrix_manipulation.cpp ex_matrix_manipulation_moddims

The [T()](\ref af::array::T) and [H()](\ref af::array::H) methods can be
used to form the [matrix or vector transpose](\ref af::array::T) .

\snippet test/matrix_manipulation.cpp ex_matrix_manipulation_transpose
