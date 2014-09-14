#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <err_cuda.hpp>

#undef _GLIBCXX_USE_INT128
#include <scan.hpp>
#include <complex>

namespace cuda
{
    template<af_op_t op, typename Ti, typename To>
    Array<To>* scan(const Array<Ti> &in, const int dim)
    {
        CUDA_NOT_SUPPORTED();
        return NULL;
    }


#define INSTANTIATE(ROp, Ti, To)                                        \
    template Array<To>* scan<ROp, Ti, To>(const Array<Ti> &in, const int dim); \

    //accum
    INSTANTIATE(af_add_t, float  , float  )
    INSTANTIATE(af_add_t, double , double )
    INSTANTIATE(af_add_t, cfloat , cfloat )
    INSTANTIATE(af_add_t, cdouble, cdouble)
    INSTANTIATE(af_add_t, int    , int    )
    INSTANTIATE(af_add_t, uint   , uint   )
    INSTANTIATE(af_add_t, char   , int    )
    INSTANTIATE(af_add_t, uchar  , uint   )
    INSTANTIATE(af_notzero_t, uchar  , uint   )
}
