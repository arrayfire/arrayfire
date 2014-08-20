#pragma once
#include <af/defines.h>
#include <af/array.h>

#define AF_MAX_DIMS 4

#ifdef __cplusplus
extern "C" {
#endif
    // Add all the elements along a dimension
    AFAPI af_err af_sum(af_array *out, const af_array in, const int dim);

    // Get the minimum of all elements along a dimension
    AFAPI af_err af_min(af_array *out, const af_array in, const int dim);

    // Get the maximum of all elements along a dimension
    AFAPI af_err af_max(af_array *out, const af_array in, const int dim);

    // Check if all elements along a dimension are true
    AFAPI af_err af_alltrue(af_array *out, const af_array in, const int dim);

    // Check if any elements along a dimension are true
    AFAPI af_err af_anytrue(af_array *out, const af_array in, const int dim);

    // Count number of non zero elements along a dimension
    AFAPI af_err af_count(af_array *out, const af_array in, const int dim);

#ifdef __cplusplus
}
#endif
