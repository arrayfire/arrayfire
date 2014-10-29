#include <af/array.h>
#include <af/arith.h>
#include "error.hpp"

namespace af
{

#define INSTANTIATE(cppfunc, cfunc)                                 \
    array cppfunc(const array &lhs, const array &rhs)               \
    {                                                               \
        af_array out = 0;                                           \
        cfunc(&out, lhs.get(), rhs.get());                          \
        return array(out);                                          \
    }

    INSTANTIATE(min, af_minof)
    INSTANTIATE(max, af_maxof)
    INSTANTIATE(pow, af_pow  )
    INSTANTIATE(rem, af_rem  )
    INSTANTIATE(mod, af_mod  )

    INSTANTIATE(cplx2, af_cplx2)
    INSTANTIATE(atan2, af_atan2)

#define WRAPPER(func)                                               \
    array func(const array &lhs, const double rhs)                  \
    {                                                               \
        return func(lhs, constant(rhs, lhs.dims(), lhs.type()));    \
    }

    WRAPPER(min)
    WRAPPER(max)
    WRAPPER(pow)
    WRAPPER(rem)
    WRAPPER(mod)
}
