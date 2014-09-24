#pragma once
#include <af/defines.h>

#ifdef __cplusplus
extern "C" {
#endif

    // Create af_array from a constant value
    AFAPI af_err af_constant(af_array *arr, const double val, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Create af_array from memory
    AFAPI af_err af_create_array(af_array *arr, const void * const data, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Create af_array handle without initializing values
    AFAPI af_err af_create_handle(af_array *arr, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Get the number of elements in an af_array
    AFAPI af_err af_get_elements(dim_type *elems, const af_array arr);

    // Get the data type of an af_array
    AFAPI af_err af_get_type(af_dtype *type, const af_array arr);

    // Copy data from an af_array to a C pointer.
    // Needs to used in conjunction with the two functions above
    AFAPI af_err af_get_data_ptr(void *data, const af_array arr);

    // Destroy af_array
    AFAPI af_err af_destroy_array(af_array arr);

    // Print contents of af_array to console
    AFAPI af_err af_print(af_array arr);

    // Create a new af_array by indexing from existing af_array.
    // This takes the form `out = in(seq_a, seq_b)`
    AFAPI af_err af_index(af_array *out, const af_array in, unsigned ndims, const af_seq* const index);

    // Tile an Array
    AFAPI af_err af_tile(af_array *out, const af_array in,
                         const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    // Reorder an Array
    AFAPI af_err af_reorder(af_array *out, const af_array in,
                            const unsigned x, const unsigned y, const unsigned z, const unsigned w);

    // Reorder an Array
    AFAPI af_err af_shift(af_array *out, const af_array in, const int x, const int y, const int z, const int w);

    // Compute first order difference along a given dimension.
    AFAPI af_err af_diff1(af_array *out, const af_array in, const int dim);

    // Compute second order difference along a given dimension.
    AFAPI af_err af_diff2(af_array *out, const af_array in, const int dim);

    // histogram: return af_array will have elements of type u32
    AFAPI af_err af_histogram(af_array *out, const af_array in, const unsigned nbins, const double minval, const double maxval);

    // re-shape the the dimensions of the input array
    AFAPI af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_type * const dims);

    // matrix transpose
    AFAPI af_err af_transpose(af_array *out, af_array in);

    // Generate Random Numbers using uniform distribution
    AFAPI af_err af_randu(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Generate Random Numbers using normal distribution
    AFAPI af_err af_randn(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Inclusive sum of all the elements along an array
    AFAPI af_err af_accum(af_array *out, const af_array in, const int dim);

    // Interpolation in 1D
    AFAPI af_err af_approx1(af_array *out, const af_array in, const af_array pos,
                            const af_interp_type method, const float offGrid);

    // Interpolation in 2D
    AFAPI af_err af_approx2(af_array *out, const af_array in, const af_array pos0, const af_array pos1,
                            const af_interp_type method, const float offGrid);

    // Sort
    AFAPI af_err af_sort(af_array *sorted, af_array *indices, const af_array in,
                         const bool dir, const unsigned dim);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace af {
    class array {
    };
}
#endif
