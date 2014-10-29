#pragma once
#include <af/array.h>

#ifdef __cplusplus
namespace af
{

AFAPI array fft(const array& in, double normalize, dim_type pad0=0);

AFAPI array fft2(const array& in, double normalize, dim_type pad0=0, dim_type pad1=0);

AFAPI array fft3(const array& in, double normalize, dim_type pad0=0, dim_type pad1=0, dim_type pad2=0);

AFAPI array ifft(const array& in, double normalize, dim_type pad0=0);

AFAPI array ifft2(const array& in, double normalize, dim_type pad0=0, dim_type pad1=0);

AFAPI array ifft3(const array& in, double normalize, dim_type pad0=0, dim_type pad1=0, dim_type pad2=0);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_fft(af_array *out, af_array in, double normalize, dim_type pad0);

    AFAPI af_err af_fft2(af_array *out, af_array in, double normalize, dim_type pad0, dim_type pad1);

    AFAPI af_err af_fft3(af_array *out, af_array in, double normalize, dim_type pad0, dim_type pad1, dim_type pad2);

    AFAPI af_err af_ifft(af_array *out, af_array in, double normalize, dim_type pad0);

    AFAPI af_err af_ifft2(af_array *out, af_array in, double normalize, dim_type pad0, dim_type pad1);

    AFAPI af_err af_ifft3(af_array *out, af_array in, double normalize, dim_type pad0, dim_type pad1, dim_type pad2);

#ifdef __cplusplus
}
#endif
