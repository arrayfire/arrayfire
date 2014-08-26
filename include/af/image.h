#pragma once
#include <af/array.h>

#ifdef __cplusplus
extern "C" {
#endif
    // Image IO: Load and Save Image functions
    AFAPI af_err af_load_image(af_array *out, const char* filename, const bool isColor);
    AFAPI af_err af_save_image(const char* filename, const af_array in);

#ifdef __cplusplus
}
#endif
