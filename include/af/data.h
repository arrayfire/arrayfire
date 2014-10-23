#pragma once
#include <af/defines.h>

#ifdef __cplusplus
extern "C" {
#endif

    // Compute first order difference along a given dimension.
    AFAPI af_err af_diff1(af_array *out, const af_array in, const int dim);

    // Compute second order difference along a given dimension.
    AFAPI af_err af_diff2(af_array *out, const af_array in, const int dim);

    // Inclusive sum of all the elements along an array
    AFAPI af_err af_accum(af_array *out, const af_array in, const int dim);

    AFAPI af_err af_where(af_array *idx, const af_array in);

    // Interpolation in 1D
    AFAPI af_err af_approx1(af_array *out, const af_array in, const af_array pos,
                            const af_interp_type method, const float offGrid);

    // Interpolation in 2D
    AFAPI af_err af_approx2(af_array *out, const af_array in, const af_array pos0, const af_array pos1,
                            const af_interp_type method, const float offGrid);

    // Sort
    AFAPI af_err af_sort(af_array *out, const af_array in, const unsigned dim, const bool dir);

    AFAPI af_err af_sort_index(af_array *out, af_array *indices, const af_array in,
                               const unsigned dim, const bool dir);

    AFAPI af_err af_sort_by_key(af_array *out_keys, af_array *out_values,
                                const af_array keys, const af_array values, const unsigned dim, const bool dir);

#ifdef __cplusplus
}
#endif
