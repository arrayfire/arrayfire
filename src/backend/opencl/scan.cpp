#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <scan.hpp>
#include <complex>
#include <stdexcept>

namespace opencl
{
    template<af_op_t op, typename Ti, typename To>
    Array<To>* scan(const Array<Ti>& in, const int dim)
    {
        throw std::runtime_error("Scan algorithms not implemented in OpenCL backend");
        Array<To> *out = createEmptyArray<To>(in.dims());
        return out;
    }

#define INSTANTIATE(ROp, Ti, To)                                        \
    template Array<To>* scan<ROp, Ti, To>(const Array<Ti>& in, const int dim); \

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
