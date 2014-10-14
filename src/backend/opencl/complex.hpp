#include <af/defines.h>
#include <af/array.h>
#include <Array.hpp>
#include <optypes.hpp>

namespace opencl
{
    template<typename To, typename Ti>
    Array<To>* complexOp(const Array<Ti> &lhs, const Array<Ti> &rhs);
}
