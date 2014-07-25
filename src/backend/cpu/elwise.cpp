#include <tuple>
#include <algorithm>
#include <map>
#include <functional>
#include <af/defines.h>
#include <af/array.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <elwise.hpp>

namespace cpu
{

using std::declval;
using std::map;
using std::swap;
using std::function;
using std::tuple;
using std::make_tuple;

template<typename T, typename U, typename Op>
void binOp(af_array *result, const af_array lhs, const af_array rhs)
{
    typedef decltype(Op()(declval<T>(), declval<U>())) ret_type;
    const Array<T> &lhs_arr = getArray<T>(lhs);
    const Array<U> &rhs_arr = getArray<U>(rhs);
    Array<ret_type> *res= createArray<ret_type>(lhs_arr.dims(), 0);

    transform(  lhs_arr.get(), lhs_arr.get() + lhs_arr.elements(),
                rhs_arr.get(),
                res->get(),
                Op());

    af_array out = getHandle(*res);
    swap(*result, out);
}

binaryOp getFunction(af_dtype lhs, af_dtype rhs)
{
    static map<tuple<af_dtype, af_dtype>, binaryOp> addFunctions;
    static map<tuple<af_dtype, af_dtype>, binaryOp> minusFunctions;
    if(addFunctions.empty()) {
        addFunctions[make_tuple(f32,f32)] = binOp<float,float, std::plus<float>>;
        addFunctions[make_tuple(f32,f64)] = binOp<float,double, std::plus<double>>;
        addFunctions[make_tuple(f64,f32)] = binOp<double,float, std::plus<double>>;
        addFunctions[make_tuple(f64,f64)] = binOp<double,double, std::plus<double>>;
    }
    return addFunctions[make_tuple(lhs,rhs)];

}
} //namespace af
