#include <algorithm>
#include <functional>
#include <assert.h>
#include <af/array.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <elwise.hpp>
#include <kernel/binaryOp.hpp>

namespace opencl
{

    using std::swap;
    using std::declval;

    template<typename Tl, typename Tr, typename To, typename Op>
    Array<To>* binOp(const Array<Tl> &lhs, const Array<Tr> &rhs)
    {
        Array<To> *res= createValueArray<To>(lhs.dims(), 0);
        kernel::binaryOp<To, Tl, Tr, Op>(res->get(),
                                         lhs.get(),
                                         rhs.get(),
                                         lhs.elements());
        return res;
    }

#define INSTANTIATE(L, R, O, OP)                                        \
    template Array<O>* binOp<L, R, O, OP>(const Array<L> &lhs, const Array<R> &rhs); \

    INSTANTIATE(float ,  float,  float, std::plus<float>);
    INSTANTIATE(float , double, double, std::plus<double>);
    INSTANTIATE(double,  float, double, std::plus<double>);
    INSTANTIATE(double, double, double, std::plus<double>);

} //namespace opencl
