#include <af/defines.h>
#include <af/array.h>
#include <Array.hpp>
#include <optypes.hpp>
#include <err_cpu.hpp>

namespace cpu
{
    template<typename T, af_op_t op>
    Array<uchar>* logicOp(const Array<T> &lhs, const Array<T> &rhs)
    {
        CPU_NOT_SUPPORTED();
    }
}
