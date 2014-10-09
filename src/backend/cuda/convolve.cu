#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <convolve.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T, typename accT, dim_type baseDim, bool expand>
Array<T> * convolve(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind)
{
    Array<T> * out = 0;
    CUDA_NOT_SUPPORTED();
    return out;
}

template<typename T, typename accT, bool expand>
Array<T> * convolve2(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter)
{
    Array<T> * out = 0;
    CUDA_NOT_SUPPORTED();
    return out;
}

#define INSTANTIATE(T, accT)  \
    template Array<T> * convolve <T, accT, 1, true >(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 1, false>(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 2, true >(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 2, false>(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 3, true >(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve <T, accT, 3, false>(Array<T> const& signal, Array<T> const& filter, ConvolveBatchKind kind);   \
    template Array<T> * convolve2<T, accT, true >(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter);  \
    template Array<T> * convolve2<T, accT, false>(Array<T> const& signal, Array<T> const& c_filter, Array<T> const& r_filter);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat ,  cfloat)
INSTANTIATE(double ,  double)
INSTANTIATE(float  ,   float)
INSTANTIATE(uint   ,   float)
INSTANTIATE(int    ,   float)
INSTANTIATE(uchar  ,   float)
INSTANTIATE(char   ,   float)

}
