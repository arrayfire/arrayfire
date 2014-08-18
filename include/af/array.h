#pragma once
#include <af/defines.h>
#include <af/dim4.hpp>
#define AF_MAX_DIMS 4

typedef long long af_array;

#ifdef __cplusplus
extern "C" {
#endif

    // Create af_array from a constant value
    AFAPI af_err af_constant(af_array *arr, const double val, const unsigned ndims, const dim_type * const dims, const af_dtype type);

    // Create af_array from memory
    AFAPI af_err af_create_array(af_array *arr, const void * const data, const unsigned ndims, const dim_type * const dims, const af_dtype type);

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
    AFAPI af_err af_index(af_array *out, af_array in, unsigned ndims, const af_seq* const index );

    // re-shape the the dimensions of the input array
    AFAPI af_err af_moddims(af_array *out, const af_array in, const unsigned ndims, const dim_type * const dims);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace af {
    class array {
    };
}
#endif
