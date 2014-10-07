#include <af/defines.h>
#include <af/array.h>
#include <Array.hpp>
#include <optypes.hpp>

namespace opencl
{
    template<typename T, af_op_t op>
    Array<T>* arithOp(const Array<T> &lhs, const Array<T> &rhs);
}
