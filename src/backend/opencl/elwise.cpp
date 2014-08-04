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
using std::string;

class plus
{
public:
    template<typename T, typename U>
    auto
    operator()(const T& lhs, const U& rhs) -> decltype(declval<T>() + declval<U>())
    {
        return lhs + rhs;
    }

    constexpr static const char * const stringOp() { return "+"; }
};

template<typename T, typename U, typename Op>
void binOp(af_array *result, const af_array lhs, const af_array rhs)
{
    typedef decltype(Op()(declval<T>(), declval<U>())) ret_type;
    const Array<T> &lhs_impl = getArray<T>(lhs);
    const Array<U> &rhs_impl = getArray<U>(rhs);
    Array<ret_type> *res= createValueArray<ret_type>(lhs_impl.dims(), 0);

    kernel::binaryOp<ret_type, T, U, Op>(   res->get(),
                                            lhs_impl.get(),
                                            rhs_impl.get(),
                                            lhs_impl.elements());

    af_array out = getHandle(*res);
    swap(*result, out);
}

    binaryOp
    getFunction(af_dtype lhs, af_dtype rhs)
    {
        switch (lhs) {
        case f32:
            switch (rhs) {
            case f32:    return binOp<float,float, plus>;
            case f64:    return binOp<float,double, plus>;
            case s32:    return binOp<float,int, plus>;
            default:     assert("NOT IMPLEMENTED" && 1 != 1);
            }
        case f64:
            switch (rhs) {
            case f32:    return binOp<double,float, plus>;
            case f64:    return binOp<double,double, plus>;
            default:     assert("NOT IMPLEMENTED" && 1 != 1);
            }
        case s32:
            switch (rhs) {
            case f32:    return binOp<int,float, plus>;
            case f64:    return binOp<int,double, plus>;
            default:     assert("NOT IMPLEMENTED" && 1 != 1);
            }
        default:     assert("NOT IMPLEMENTED" && 1 != 1);
        }
        return NULL;
    }

} //namespace opencl
