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

\democode{
float h[] = {1, 2, 3, 4};
array small = array(2, 2, h); // 2x2 matrix
af_print(small);
array large = tile(small, 2, 3);  // produces 8x12 matrix: (2*2)x(2*3)
}

join() allows you to joining two matrices together.  Matrix
dimensions must match along every dimension except the dimension
of joining (dimensions are 0-indexed). For example, a 2x3 matrix
can be joined with a 2x4 matrix along dimension 1, but not along
dimension 0 since {3,4} don`t match up.

\democode{
float hA[] = { 1, 2, 3, 4, 5, 6 };
float hB[] = { 10, 20, 30, 40, 50, 60, 70, 80, 90 };
array A = array(3, 2, hA);
array B = array(3, 3, hB);

af_print(join(1, A, B)); // 3x5 matrix
// array result = join(0, A, B); // fail: dimension mismatch
}

Construct a regular mesh grid from vectors `x` and `y`. For example, a
mesh grid of the vectors {1,2,3,4} and {5,6} would result in two matrices:

\democode{
float hx[] = {1, 2, 3, 4};
float hy[] = {5, 6};

array x = array(4, hx);
array y = array(2, hy);

af_print(tile(x, 1, 2));
af_print(tile(y.T(), 4, 1));
}

[array()](\ref af::array) can be used to create a (shallow) copy of a matrix
with different dimensions.  The number of elements must remain the same as
the original array.

\democode{
int hA[] = {1, 2, 3, 4, 5, 6};
array A = array(3, 2, hA);

af_print(array(A, 2, 3)); // 2x3 matrix
af_print(array(A, 6, 1)); // 6x1 column vector

// af_print(array(A, 2, 2)); // fail: wrong number of elements
// af_print(array(A, 8, 8)); // fail: wrong number of elements
}

The [T()](\ref af::array::T) and [H()](\ref af::array::H) methods can be
used to form the [matrix or vector transpose](\ref af::array::T) .

\democode{
array x = randu(2, 2, f64);
af_print(x.T());  // transpose (real)

array c = randu(2, 2, c64);
af_print(c.T());  // transpose (complex)
af_print(c.H());  // Hermitian (conjugate) transpose
}
