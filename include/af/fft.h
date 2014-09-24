#pragma once
#include <af/array.h>
#include <af/defines.h>

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_fft(af_array *out, af_array in, af_fft_kind kind,
                        double normalize, dim_type pad0);

    AFAPI af_err af_fft2(af_array *out, af_array in, af_fft_kind kind,
                         double normalize, dim_type pad0, dim_type pad1);

    AFAPI af_err af_fft3(af_array *out, af_array in, af_fft_kind kind,
                         double normalize, dim_type pad0, dim_type pad1, dim_type pad2);

    AFAPI af_err af_ifft(af_array *out, af_array in, af_fft_kind kind,
                         double normalize, dim_type pad0);

    AFAPI af_err af_ifft2(af_array *out, af_array in, af_fft_kind kind,
                          double normalize, dim_type pad0, dim_type pad1);

    AFAPI af_err af_ifft3(af_array *out, af_array in, af_fft_kind kind,
                          double normalize, dim_type pad0, dim_type pad1, dim_type pad2);

#ifdef __cplusplus
}
#endif
