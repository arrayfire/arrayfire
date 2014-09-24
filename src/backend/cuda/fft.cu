#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <fft.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType> * fft(Array<inType> const &in, double normalize, dim_type const npad, dim_type const * const pad)
{
    Array<outType> *out = 0;
    CUDA_NOT_SUPPORTED();
    return out;
}

template<typename inType, typename outType, int rank>
Array<outType> * ifft(Array<inType> const &in, double normalize, dim_type const npad, dim_type const * const pad)
{
    Array<outType> *out = 0;
    CUDA_NOT_SUPPORTED();
    return out;
}

#define INSTANTIATE1(T1, T2)\
    template Array<T2> * fft <T1, T2, 1, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 2, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 3, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 1, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 2, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 3, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE1(float  , cfloat )
INSTANTIATE1(double , cdouble)

#define INSTANTIATE2(T1, T2)\
    template Array<T2> * fft <T1, T2, 1, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 2, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 3, false>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * ifft<T1, T2, 1>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * ifft<T1, T2, 2>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad);\
    template Array<T2> * ifft<T1, T2, 3>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE2(cfloat , cfloat )
INSTANTIATE2(cdouble, cdouble)

#define INSTANTIATE3(T1, T2)\
    template Array<T2> * ifft<T1, T2, 1>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * ifft<T1, T2, 2>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * ifft<T1, T2, 3>(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE3(cfloat , float )
INSTANTIATE3(cdouble, double)

}
