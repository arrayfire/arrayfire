#include "reduce_impl.hpp"

namespace opencl
{
    // count
    INSTANTIATE(af_notzero_t, float  , uint)
    INSTANTIATE(af_notzero_t, double , uint)
    INSTANTIATE(af_notzero_t, cfloat , uint)
    INSTANTIATE(af_notzero_t, cdouble, uint)
    INSTANTIATE(af_notzero_t, int    , uint)
    INSTANTIATE(af_notzero_t, uint   , uint)
    INSTANTIATE(af_notzero_t, char   , uint)
    INSTANTIATE(af_notzero_t, uchar  , uint)
}
