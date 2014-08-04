#include <algorithm>
#include <functional>
#include <boost/utility/declval.hpp>
#include <boost/typeof/typeof.hpp>
#include <af/array.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <elwise.hpp>
#include <kernel/elwise.hpp>


namespace cuda
{

using std::swap;
using boost::declval;

template<typename T, typename U, typename Op>
void binOp(af_array *result, const af_array lhs, const af_array rhs)
{
    typedef BOOST_TYPEOF(Op()(declval<T>(), declval<U>())) ret_type;
    const Array<T> &lhs_arr = getArray<T>(lhs);
    const Array<U> &rhs_arr = getArray<U>(rhs);
    Array<ret_type> *res= createValueArray<ret_type>(lhs_arr.dims(), 0);

    kernel::binaryOp(res->get(), lhs_arr.get(), rhs_arr.get(), lhs_arr.elements());

    af_array out = getHandle(*res);
    swap(*result, out);
}

binaryOp getFunction(af_dtype lhs, af_dtype rhs)
{
    switch (lhs) {
        case f32:
            switch (rhs) {
                case f32:    return binOp<float,float, std::plus<float> >;
                case f64:    return binOp<float,double, std::plus<double> >;
            }
        case f64:
            switch (rhs) {
                case f32:    return binOp<double,float, std::plus<double> >;
                case f64:    return binOp<double,double, std::plus<double> >;
            }
    }
    return NULL;
}

} //namespace cuda
