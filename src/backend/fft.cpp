#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/fft.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <fft.hpp>

using af::dim4;
using namespace detail;

template<typename inType, typename outType, int rank, bool isR2C>
static af_array fft(af_array in, double normalize, dim_type const npad, dim_type const * const pad)
{
    return getHandle(*fft<inType, outType, rank, isR2C>(getArray<inType>(in), normalize, npad, pad));
}

template<typename inType, typename outType, int rank>
static af_array ifft(af_array in, double normalize, dim_type const npad, dim_type const * const pad)
{
    return getHandle(*ifft<inType, outType, rank>(getArray<inType>(in), normalize, npad, pad));
}

template<int rank>
static af_err fft(af_array *out, af_array in, af_fft_kind kind, double normalize, dim_type npad, dim_type const * const pad)
{
    try {
        ARG_ASSERT(2, (kind!=AF_C2R));

        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 dims  = info.dims();

        DIM_ASSERT(1, (dims.ndims()>=rank));
        DIM_ASSERT(1, (dims.ndims()<=(rank+1)));

        af_array output;
        switch(kind) {
            case AF_C2C:
                {
                    switch(type) {
                        case c32: output = fft<cfloat , cfloat , rank, false>(in, normalize, npad, pad); break;
                        case c64: output = fft<cdouble, cdouble, rank, false>(in, normalize, npad, pad); break;
                        case f32: output = fft<float , cfloat  , rank, true >(in, normalize, npad, pad); break;
                        case f64: output = fft<double, cdouble , rank, true >(in, normalize, npad, pad); break;
                        default: TYPE_ERROR(1, type);
                    }
                }
                break;
            case AF_R2C:
                {
                    switch(type) {
                        case f32: output = fft<float  , cfloat , rank, false>(in, normalize, npad, pad); break;
                        case f64: output = fft<double , cdouble, rank, false>(in, normalize, npad, pad); break;
                        default: TYPE_ERROR(1, type);
                    }
                }
                break;
            default:
                ARG_ASSERT(2, (kind>=AF_R2C && kind<=AF_C2C));
                break;
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

template<int rank>
static af_err ifft(af_array *out, af_array in, af_fft_kind kind, double normalize, dim_type npad, dim_type const * const pad)
{
    try {
        ARG_ASSERT(2, kind!=AF_R2C);

        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 dims  = info.dims();

        DIM_ASSERT(1, (dims.ndims()>=rank));
        DIM_ASSERT(1, (dims.ndims()<=(rank+1)));

        af_array output;

        switch(kind) {
            case AF_C2R:
                {
                    switch(type) {
                        case c32: output = ifft<cfloat , float , rank>(in, normalize, npad, pad); break;
                        case c64: output = ifft<cdouble, double, rank>(in, normalize, npad, pad); break;
                        default: TYPE_ERROR(1, type);
                    }
                }
                break;
            case AF_C2C:
                {
                    switch(type) {
                        case c32: output = ifft<cfloat , cfloat , rank>(in, normalize, npad, pad); break;
                        case c64: output = ifft<cdouble, cdouble, rank>(in, normalize, npad, pad); break;
                        default: TYPE_ERROR(1, type);
                    }
                }
                break;
            default:
                ARG_ASSERT(2, (kind>=AF_R2C && kind<=AF_C2C));
                break;
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fft(af_array *out, af_array in, af_fft_kind kind, double normalize, dim_type pad0)
{
    dim_type pad[1] = {pad0};
    return fft<1>(out, in, kind, normalize, (pad0>0?1:0), pad);
}

af_err af_fft2(af_array *out, af_array in, af_fft_kind kind, double normalize, dim_type pad0, dim_type pad1)
{
    dim_type pad[2] = {pad0, pad1};
    return fft<2>(out, in, kind, normalize, (pad0>0&&pad1>0?2:0), pad);
}

af_err af_fft3(af_array *out, af_array in, af_fft_kind kind, double normalize, dim_type pad0, dim_type pad1, dim_type pad2)
{
    dim_type pad[3] = {pad0, pad1, pad2};
    return fft<3>(out, in, kind, normalize, (pad0>0&&pad1>0&&pad2>0?3:0), pad);
}

af_err af_ifft(af_array *out, af_array in, af_fft_kind kind, double normalize, dim_type pad0)
{
    dim_type pad[1] = {pad0};
    return ifft<1>(out, in, kind, normalize, (pad0>0?1:0), pad);
}

af_err af_ifft2(af_array *out, af_array in, af_fft_kind kind, double normalize, dim_type pad0, dim_type pad1)
{
    dim_type pad[2] = {pad0, pad1};
    return ifft<2>(out, in, kind, normalize, (pad0>0&&pad1>0?2:0), pad);
}

af_err af_ifft3(af_array *out, af_array in, af_fft_kind kind, double normalize, dim_type pad0, dim_type pad1, dim_type pad2)
{
    dim_type pad[3] = {pad0, pad1, pad2};
    return ifft<3>(out, in, kind, normalize, (pad0>0&&pad1>0&&pad2>0?3:0), pad);
}
