#include <complex>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <reduce.hpp>
#include <err_opencl.hpp>

using std::swap;
using af::dim4;
namespace opencl
{
    template<af_op_t op, typename Ti, typename To>
    Array<To>* reduce(const Array<Ti> &in, const int dim)
    {
        OPENCL_NOT_SUPPORTED();
        return NULL;
    }

#define INSTANTIATE(Op, Ti, To)                                         \
    template Array<To>* reduce<Op, Ti, To>(const Array<Ti> &in, const int dim); \

    //min
    INSTANTIATE(af_min_t, float  , float  )
    INSTANTIATE(af_min_t, double , double )
    INSTANTIATE(af_min_t, cfloat , cfloat )
    INSTANTIATE(af_min_t, cdouble, cdouble)
    INSTANTIATE(af_min_t, int    , int    )
    INSTANTIATE(af_min_t, uint   , uint   )
    INSTANTIATE(af_min_t, char   , char   )
    INSTANTIATE(af_min_t, uchar  , uchar  )

    //max
    INSTANTIATE(af_max_t, float  , float  )
    INSTANTIATE(af_max_t, double , double )
    INSTANTIATE(af_max_t, cfloat , cfloat )
    INSTANTIATE(af_max_t, cdouble, cdouble)
    INSTANTIATE(af_max_t, int    , int    )
    INSTANTIATE(af_max_t, uint   , uint   )
    INSTANTIATE(af_max_t, char   , char   )
    INSTANTIATE(af_max_t, uchar  , uchar  )

    //sum
    INSTANTIATE(af_add_t, float  , float  )
    INSTANTIATE(af_add_t, double , double )
    INSTANTIATE(af_add_t, cfloat , cfloat )
    INSTANTIATE(af_add_t, cdouble, cdouble)
    INSTANTIATE(af_add_t, int    , int    )
    INSTANTIATE(af_add_t, uint   , uint   )
    INSTANTIATE(af_add_t, char   , int    )
    INSTANTIATE(af_add_t, uchar  , uint   )

    // count
    INSTANTIATE(af_notzero_t, float  , uint)
    INSTANTIATE(af_notzero_t, double , uint)
    INSTANTIATE(af_notzero_t, cfloat , uint)
    INSTANTIATE(af_notzero_t, cdouble, uint)
    INSTANTIATE(af_notzero_t, int    , uint)
    INSTANTIATE(af_notzero_t, uint   , uint)
    INSTANTIATE(af_notzero_t, char   , uint)
    INSTANTIATE(af_notzero_t, uchar  , uint)

    //anytrue
    INSTANTIATE(af_or_t, float  , uchar)
    INSTANTIATE(af_or_t, double , uchar)
    INSTANTIATE(af_or_t, cfloat , uchar)
    INSTANTIATE(af_or_t, cdouble, uchar)
    INSTANTIATE(af_or_t, int    , uchar)
    INSTANTIATE(af_or_t, uint   , uchar)
    INSTANTIATE(af_or_t, char   , uchar)
    INSTANTIATE(af_or_t, uchar  , uchar)

    //alltrue
    INSTANTIATE(af_and_t, float  , uchar)
    INSTANTIATE(af_and_t, double , uchar)
    INSTANTIATE(af_and_t, cfloat , uchar)
    INSTANTIATE(af_and_t, cdouble, uchar)
    INSTANTIATE(af_and_t, int    , uchar)
    INSTANTIATE(af_and_t, uint   , uchar)
    INSTANTIATE(af_and_t, char   , uchar)
    INSTANTIATE(af_and_t, uchar  , uchar)
}
