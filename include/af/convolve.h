#pragma once
#include <af/array.h>

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_convolve1(af_array *out, af_array signal, af_array filter, bool expand);

    AFAPI af_err af_convolve2(af_array *out, af_array signal, af_array filter, bool expand);

    AFAPI af_err af_convolve3(af_array *out, af_array signal, af_array filter, bool expand);

    AFAPI af_err af_convolve2_sep(af_array *out, af_array signal, af_array col_filter, af_array row_filter, bool expand);

#ifdef __cplusplus
}
#endif
