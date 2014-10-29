#include <af/array.h>
#include "error.hpp"

namespace af
{
//#define INSTANTIATE(op, func)                                           \
//    array operator##op(const array &lhs, const array &rhs)              \
//    {                                                                   \
//        af_array out = 0;                                               \
//        af_##func(&out, lhs.get(), rhs.get());                          \
//        return array(out);                                              \
//    }
//
//    INSTANTIATE(+, add)
//    INSTANTIATE(-, sub)
//    INSTANTIATE(*, mul)
//    INSTANTIATE(/, div)
//
//#define INSTANTIATE1(op, func)                                          \
//    template<typename T>                                                \
//    array operator##op(const array &lhs, const T rhs)                   \
//    {                                                                   \
//        af_array out = 0;                                               \
//        array rhs_ = constant(rhs, lhs.dims(), lhs.type());             \
//        af_##func(&out, lhs.get(), rhs_.get());                         \
//        return array(out);                                              \
//    }
//
//#define INITTYPE(type)                                                  \
//    INSTANTIATE2(+, add)                                                \
//    INSTANTIATE2(-, sub)                                                \
//    INSTANTIATE2(*, mul)                                                \
//    INSTANTIATE2(/, div)                                                \
//
//
//    INITTYPE(af_cdouble)
//    INITTYPE(af_cfloat)
//    INITTYPE(double)
//    INITTYPE(float)
//    INITTYPE(unsigned)
//    INITTYPE(int)
//    INITTYPE(unsigned char)
//    INITTYPE(char)
}
