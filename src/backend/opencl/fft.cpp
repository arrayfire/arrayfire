#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <fft.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType> * fft(Array<inType> const &in, double normalize, dim_type const npad, dim_type const * const pad)
{
    Array<outType> *out = nullptr;
    OPENCL_NOT_SUPPORTED();
    return out;
}

template<typename T, int rank>
Array<T> * ifft(Array<T> const &in, double normalize, dim_type const npad, dim_type const * const pad)
{
    Array<T> *out = nullptr;
    OPENCL_NOT_SUPPORTED();
    return out;
}

#define INSTANTIATE1(T1, T2)\
    template Array<T2> * fft <T1, T2, 1, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 2, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 3, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE1(float  , cfloat )
INSTANTIATE1(double , cdouble)

#define INSTANTIATE2(T)\
    template Array<T> * fft <T, T, 1, false>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * fft <T, T, 2, false>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * fft <T, T, 3, false>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * ifft<T, 1>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * ifft<T, 2>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * ifft<T, 3>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE2(cfloat )
INSTANTIATE2(cdouble)

}
