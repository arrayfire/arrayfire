#include <af/defines.h>
#include <af/array.h>
#include <Array.hpp>
#include <optypes.hpp>
#include <err_cuda.hpp>
#include <binary.hpp>

namespace cuda
{
    template<typename T, af_op_t op>
    Array<T>* arithOp(const Array<T> &lhs, const Array<T> &rhs)
    {
        return createBinaryNode<T, T, op>(lhs, rhs);
    }
}
