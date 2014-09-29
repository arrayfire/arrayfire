#include "reduce_impl.hpp"

namespace opencl
{
    //anytrue
    INSTANTIATE(af_or_t, float  , uchar)
    INSTANTIATE(af_or_t, double , uchar)
    INSTANTIATE(af_or_t, cfloat , uchar)
    INSTANTIATE(af_or_t, cdouble, uchar)
    INSTANTIATE(af_or_t, int    , uchar)
    INSTANTIATE(af_or_t, uint   , uchar)
    INSTANTIATE(af_or_t, char   , uchar)
    INSTANTIATE(af_or_t, uchar  , uchar)
}
