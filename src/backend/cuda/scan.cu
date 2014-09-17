#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <err_cuda.hpp>

#undef _GLIBCXX_USE_INT128
#include <scan.hpp>
#include <complex>
#include <kernel/scan_first.hpp>
#include <kernel/scan_dim.hpp>

namespace cuda
{
    template<af_op_t op, typename Ti, typename To>
    Array<To>* scan(const Array<Ti> &in, const int dim)
    {
        Array<To> *out = createEmptyArray<To>(in.dims());

        switch (dim) {
        case 0: kernel::scan_first<Ti, To, op   >(*out, in); break;
        case 1: kernel::scan_dim  <Ti, To, op, 1>(*out, in); break;
        case 2: kernel::scan_dim  <Ti, To, op, 2>(*out, in); break;
        case 3: kernel::scan_dim  <Ti, To, op, 3>(*out, in); break;
        }

        return out;
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
