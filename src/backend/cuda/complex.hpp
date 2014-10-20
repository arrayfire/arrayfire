#include <af/defines.h>
#include <af/array.h>
#include <Array.hpp>
#include <optypes.hpp>
#include <err_cuda.hpp>

namespace cuda
{
    template<typename To, typename Ti>
    Array<To>* complexOp(const Array<Ti> &lhs, const Array<Ti> &rhs)
    {
        return createValueArray<To>(lhs.dims(), scalar<To>(0));
    }
}
