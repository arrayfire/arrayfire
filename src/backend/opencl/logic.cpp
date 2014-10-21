#include <af/dim4.hpp>
#include <af/defines.h>
#include <Array.hpp>
#include <binary.hpp>
#include <logic.hpp>
#include <complex>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T, af_op_t op>
    Array<uchar> *logicOp(const Array<T> &lhs, const Array<T> &rhs)
    {
        return createBinaryNode<uchar, T, op>(lhs, rhs);
    }

#define INSTANTIATE(T)                                                  \
    template Array<uchar>* logicOp<T, af_eq_t >(const Array<T> &lhs, const Array<T> &rhs); \
    template Array<uchar>* logicOp<T, af_neq_t>(const Array<T> &lhs, const Array<T> &rhs); \
    template Array<uchar>* logicOp<T, af_gt_t >(const Array<T> &lhs, const Array<T> &rhs); \
    template Array<uchar>* logicOp<T, af_ge_t >(const Array<T> &lhs, const Array<T> &rhs); \
    template Array<uchar>* logicOp<T, af_lt_t >(const Array<T> &lhs, const Array<T> &rhs); \
    template Array<uchar>* logicOp<T, af_le_t >(const Array<T> &lhs, const Array<T> &rhs); \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
