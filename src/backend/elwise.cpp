#include <af/array.h>
#include <af/defines.h>
#include <af/arith.h>
#include <ArrayInfo.hpp>
#include <elwise.hpp>
#include <helper.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <utility>
#include <map>
#include <boost/utility/declval.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/function.hpp>
using namespace detail;

using boost::declval;
using boost::function;
using std::map;
using std::swap;
using std::pair;
using std::make_pair;

typedef void(*binaryOp)(af_array*, const af_array, const af_array);

template<typename Tl, typename Tr, typename To, typename op>
void binOp(af_array *result, const af_array lhs, const af_array rhs)
{
    const Array<Tl> left  = getArray<Tl>(lhs);
    const Array<Tr> right = getArray<Tr>(rhs);
    Array<To>* out = detail::binOp<Tl, Tr, To, op>(left, right);
    *result = getHandle(*out);
}

#define BINOP(TL, TR, OP) binOp<TL, TR, BOOST_TYPEOF(OP()(declval<TL>(), declval<TR>())), OP >

static binaryOp getFunction(af_dtype lhs, af_dtype rhs)
{
    static map<pair<af_dtype, af_dtype>, binaryOp> addFunctions;

    if(addFunctions.empty()) {
        addFunctions[make_pair(f32,f32)] = BINOP(float, float, std::plus<float>);
        addFunctions[make_pair(f32,f64)] = BINOP(float, double, std::plus<double>);
        addFunctions[make_pair(f64,f32)] = BINOP(double, float, std::plus<double>);
        addFunctions[make_pair(f64,f64)] = BINOP(double, double, std::plus<double>);
    }
    return addFunctions[make_pair(lhs,rhs)];
}

af_err af_add(af_array *result, const af_array lhs, const af_array rhs)
{
    af_err ret = AF_SUCCESS;
    try {
        af_dtype lhs_t, rhs_t;
        af_get_type(&lhs_t, lhs);
        af_get_type(&rhs_t, rhs);
        getFunction(lhs_t , rhs_t)(result, lhs, rhs);
    }
    CATCHALL

    return ret;
}
