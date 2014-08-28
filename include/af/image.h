#pragma once
#include <af/array.h>

#ifdef __cplusplus
extern "C" {
#endif
    // Image IO: Load and Save Image functions
    AFAPI af_err af_load_image(af_array *out, const char* filename, const bool isColor);

    AFAPI af_err af_save_image(const char* filename, const af_array in);

    // Resize an image/matrix/array
    AFAPI af_err af_resize(af_array *out, const af_array in, const dim_type odim0, const dim_type odim1, const af_interp_type method);
    // image dilation operation
    AFAPI af_err af_dilate(af_array *out, const af_array in, const af_array mask);

    AFAPI af_err af_dilate3d(af_array *out, const af_array in, const af_array mask);

    // image erosion operation
    AFAPI af_err af_erode(af_array *out, const af_array in, const af_array mask);

    AFAPI af_err af_erode3d(af_array *out, const af_array in, const af_array mask);

#ifdef __cplusplus
}
#endif
