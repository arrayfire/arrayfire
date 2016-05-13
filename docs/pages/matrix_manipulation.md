Array and Matrix Manipulation {#matrixmanipulation}
===================

ArrayFire provides several different methods for
[manipulating arrays and matrices](\ref manip_mat). The functionality includes:

* moddims() - change the dimensions of an array without changing the data
* array() - create a (shallow) copy of an array with different dimensions.
* flat() - flatten an array to one dimension
* flip() - flip an array along a dimension
* join() - join up to 4 arrays
* reorder() - changes the dimension order within the array
* shift() - shifts data along a dimension
* tile() - repeats an array along a dimension
* transpose() - performs a matrix transpose
* [T()](\ref af::array::T) - transpose a matrix or vector (shorthand notation)
* [H()](\ref af::array::H) - Hermitian Transpose (conjugate-transpose) a matrix

Below we provide several examples of these functions and their use.

## flat()

The __flat()__ function flattens an array to one dimension:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
a [3 3 1 1]
    1.0000     4.0000     7.0000
    2.0000     5.0000     8.0000
    3.0000     6.0000     9.0000

flat(a) [9 1 1 1]
    1.0000
    2.0000
    3.0000
    4.0000
    5.0000
    6.0000
    7.0000
    8.0000
    9.0000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The flat function can be called from C and C++ as follows:

> __af_err af_flat(af_array* out, const af_array in)__
> --  C interface for flat() function

> __array af::flat(const array& in)__
> --  C++ interface for flat() function

## flip()

The __flip()__ function flips the contents of an array along a chosen dimension.
In the example below, we show the 5x2 array flipped along the zeroth (i.e.
within a column) and first (e.g. across rows) axes:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
a [5 2 1 1]
    1.0000     6.0000
    2.0000     7.0000
    3.0000     8.0000
    4.0000     9.0000
    5.0000    10.0000

flip(a, 0) [5 2 1 1]
    5.0000    10.0000
    4.0000     9.0000
    3.0000     8.0000
    2.0000     7.0000
    1.0000     6.0000

flip(a, 1) [5 2 1 1]
    6.0000     1.0000
    7.0000     2.0000
    8.0000     3.0000
    9.0000     4.0000
   10.0000     5.0000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The flip function can be called from C and C++ as follows:

> __af_err af_flip(af_array *out, const af_array in, const unsigned dim)__
> --  C interface for flip()

> __array af::flip(const array &in, const unsigned dim)__
> --  C++ interface for flip()

## join()

The __join()__ function joins arrays along a specific dimension. The C++
interface can join up to four arrays whereas the C interface supports up to 10
arrays. Here is an example of how to use join an array to itself:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
a [5 1 1 1]
    1.0000
    2.0000
    3.0000
    4.0000
    5.0000

join(0, a, a) [10 1 1 1]
    1.0000
    2.0000
    3.0000
    4.0000
    5.0000
    1.0000
    2.0000
    3.0000
    4.0000
    5.0000

join(1, a, a) [5 2 1 1]
    1.0000     1.0000
    2.0000     2.0000
    3.0000     3.0000
    4.0000     4.0000
    5.0000     5.0000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The join function has several candidate functions in C:

> __af_err af_join(af_array *out, const int dim, const af_array first, const af_array second)__
> --  C interface function to join 2 arrays along a dimension

> __af_err af_join_many(af_array *out, const int dim, const unsigned n_arrays, const af_array *inputs)__
> --  C interface function to join up to 10 arrays along a dimension

and in C++:

> __array af::join(const int dim, const array &first, const array &second)__
> --  Joins 2 arrays along a dimension

> __array af::join(const int dim, const array &first, const array &second, const array &third)__
> --  Joins 3 arrays along a dimension.

> __array af::join(const int dim, const array &first, const array &second, const array &third, const array &fourth)__
> --  Joins 4 arrays along a dimension


## moddims()

The __moddims()__ function changes the dimensions of an array without changing
its data or order. Note that this function modifies only the _metadata_
associated with the array. It does not modify the content of the array.
Here is an example of moddims() converting an 8x1 array into a 2x4 and then
back to a 8x1:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
a [8 1 1 1]
    1.0000
    2.0000
    1.0000
    2.0000
    1.0000
    2.0000
    1.0000
    2.0000

af::dim4 new_dims(2, 4);
moddims(a, new_dims) [2 4 1 1]
    1.0000     1.0000     1.0000     1.0000
    2.0000     2.0000     2.0000     2.0000

moddims(a, a.elements(), 1, 1, 1) [8 1 1 1]
    1.0000
    2.0000
    1.0000
    2.0000
    1.0000
    2.0000
    1.0000
    2.0000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The moddims function has a single form in the C API:

> __af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_t *const dims)__
> --  C interface to mod dimensions of an array

And several overloaded candidates in the C++ API:

> __array af::moddims(const array &in, const unsigned ndims, const dim_t *const dims)__
> --  mods number of dimensions to match _ndims_ as specidied in the array _dims_

> __array af::moddims(const array &in, const dim4 &dims)__
> --  mods dimensions as specified by _dims_

> __array af::moddims(const array &in, const dim_t d0, const dim_t d1=1, const dim_t d2=1, const dim_t d3=1)__
> --  mods dimensions of an array

## reorder()

The __reorder()__ function modifies the order of data within an array by
exchanging data according to the change in dimensionality. The linear ordering
of data within the array is preserved.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
a [2 2 3 1]
    1.0000     3.0000
    2.0000     4.0000

    1.0000     3.0000
    2.0000     4.0000

    1.0000     3.0000
    2.0000     4.0000


reorder(a, 1, 0, 2) [2 2 3 1]  //equivalent to a transpose
    1.0000     2.0000
    3.0000     4.0000

    1.0000     2.0000
    3.0000     4.0000

    1.0000     2.0000
    3.0000     4.0000


reorder(a, 2, 0, 1) [3 2 2 1]
    1.0000     2.0000
    1.0000     2.0000
    1.0000     2.0000

    3.0000     4.0000
    3.0000     4.0000
    3.0000     4.0000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reorder function has several candidates functions in the C/C++ APIs:

> __af_err af_reorder(af_array *out, const af_array in, const unsigned x, const unsigned y, const unsigned z, const unsigned w)__
> --  C interface for reordering function

> __array af::reorder(const array &in, const unsigned x, const unsigned y=1, const unsigned z=2, const unsigned w=3)__
> --  Reorders dimensions of an array

## shift()

The __shift()__ function shifts data in a circular buffer fashion along a
chosen dimension. Consider the following example:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
a [3 5 1 1]
    0.0000     0.0000     0.0000     0.0000     0.0000
    3.0000     4.0000     5.0000     1.0000     2.0000
    3.0000     4.0000     5.0000     1.0000     2.0000

shift(a, 0, 2 ) [3 5 1 1]
    0.0000     0.0000     0.0000     0.0000     0.0000
    1.0000     2.0000     3.0000     4.0000     5.0000
    1.0000     2.0000     3.0000     4.0000     5.0000

shift(a, -1, 2 ) [3 5 1 1]
    1.0000     2.0000     3.0000     4.0000     5.0000
    1.0000     2.0000     3.0000     4.0000     5.0000
    0.0000     0.0000     0.0000     0.0000     0.0000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shift function can be called from C and C++ as follows:


> __af_err af_shift(af_array *out, const af_array in, const int x, const int y, const int z, const int w)__
> --  C interface for shifting an array

> __array af::shift(const array &in, const int x, const int y=0, const int z=0, const int w=0)__
> --  Shifts array along specified dimensions

## tile()

The __tile()__ function repeats an array along the specified dimension.
For example below we show how to tile an array along the zeroth and first
dimensions of an array:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
a [3 1 1 1]
    1.0000
    2.0000
    3.0000

// Repeat array a twice in the zeroth dimension
tile(a, 2) [6 1 1 1]
    1.0000
    2.0000
    3.0000
    1.0000
    2.0000
    3.0000

// Repeat array a twice along both the zeroth and first dimensions
tile(a, 2, 2) [6 2 1 1]
    1.0000     1.0000
    2.0000     2.0000
    3.0000     3.0000
    1.0000     1.0000
    2.0000     2.0000
    3.0000     3.0000

// Repeat array a twice along the first and three times along the second
// dimension.
af::dim4 tile_dims(1, 2, 3);
tile(a, tile_dims) [3 2 3 1]
    1.0000     1.0000
    2.0000     2.0000
    3.0000     3.0000

    1.0000     1.0000
    2.0000     2.0000
    3.0000     3.0000

    1.0000     1.0000
    2.0000     2.0000
    3.0000     3.0000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The C interface for tile is as follows:

> __af_err af_tile(af_array *out, const af_array in, const unsigned x, const unsigned y, const unsigned z, const unsigned w)__
> --  C interface for tiling an array

The C++ interface has two overloads

> __array af::tile(const array &in, const unsigned x, const unsigned y=1, const unsigned z=1, const unsigned w=1)__
> --  Tiles array along specified dimensions

> __array af::tile(const array &in, const dim4 &dims)__
> --  Tile an array according to a dim4 object

## transpose()

The __transpose()__ function performs a standard matrix transpose. The input
array must have the dimensions of a 2D-matrix.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
a [3 3 1 1]
    1.0000     3.0000     3.0000
    2.0000     1.0000     3.0000
    2.0000     2.0000     1.0000

transpose(a) [3 3 1 1]
    1.0000     2.0000     2.0000
    3.0000     1.0000     2.0000
    3.0000     3.0000     1.0000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The C interfaces for transpose are as follows:

> __af_err af_transpose(af_array *out, af_array in, const bool conjugate)__
> --   C interface to transpose a matrix.

> __af_err af_transpose_inplace(af_array in, const bool conjugate)__
> --   C interface to transpose a matrix in-place.

The C++ interface has two primary functions and two shorthand versions:

> __array af::transpose(const array &in, const bool conjugate=false)__
> --   Transposes a matrix.

> __void af::transposeInPlace(array &in, const bool conjugate=false)__
> --   Transposes a matrix in-place.

> __array af::T()
> --   Transpose a matrix

> __array af::H()
> --   Conjugate Transpose (Hermitian transpose) of a matrix

Here is an example of how the shorthand versions might be used:

\snippet test/matrix_manipulation.cpp ex_matrix_manipulation_transpose

## array()

[array()](\ref af::array) can be used to create a (shallow) copy of a matrix
with different dimensions. The total number of elements must remain the same.
This function is a wrapper over the moddims() function discussed earlier.

# Combining re-ordering functions to enumerate grid coordinates

By using a combination of the array restructuring functions, one can quickly code
complex manipulation patterns with a few lines of code. For example, consider
generating (*x,y*) coordinates for a grid where each axis goes from *1 to n*.
Instead of using several loops to populate our arrays we can just use a small
combination of the above functions.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
unsigned n=3;
af::array xy = join(1,
                tile(seq(1, n), n),
                flat( transpose(tile(seq(1, n), 1, n)) )
                   );
xy [9 2 1 1]
    1.0000     1.0000
    2.0000     1.0000
    3.0000     1.0000
    1.0000     2.0000
    2.0000     2.0000
    3.0000     2.0000
    1.0000     3.0000
    2.0000     3.0000
    3.0000     3.0000
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
