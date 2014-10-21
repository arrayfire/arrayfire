#include <af/dim4.hpp>
#include <af/defines.h>
#include <Array.hpp>
#include <binary.hpp>
#include <arith.hpp>
#include <complex>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename To, typename Ti>
    Array<To>* complexOp(const Array<Ti> &lhs, const Array<Ti> &rhs)
    {
        return createBinaryNode<To, Ti, af_cplx2_t>(lhs, rhs);
    }

#define INSTANTIATE(To, Ti)                                             \
    template Array<To>* complexOp<To, Ti>(const Array<Ti> &lhs, const Array<Ti> &rhs);

    INSTANTIATE(cfloat, float)
    INSTANTIATE(cdouble, double)

}
