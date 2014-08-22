#pragma once
#include <backend.hpp>
#include <cassert>
#include <limits>

#ifndef __DH__
#define __DH__
#endif

typedef enum {
    af_add_t = 0,
    af_min_t = 1,
    af_max_t = 2,
    af_and_t = 3,
    af_or_t = 4,
    af_notzero_t = 5,
} af_op_t;

using detail::cfloat;
using detail::cdouble;

namespace ops
{

    template<typename T>
    static __DH__ T max(T lhs, T rhs)
    {
        return std::max(lhs, rhs);
    }

    template<>
    __DH__ cfloat max<cfloat>(cfloat lhs, cfloat rhs)
    {
        return detail::abs(lhs) > detail::abs(rhs) ? lhs : rhs;
    }

    template<>
    __DH__ cdouble max<cdouble>(cdouble lhs, cdouble rhs)
    {
        return detail::abs(lhs) > detail::abs(rhs) ? lhs : rhs;
    }

    template<typename T>
    static __DH__ T min(T lhs, T rhs)
    {
        return std::min(lhs, rhs);
    }

    template<>
    __DH__ cfloat min<cfloat>(cfloat lhs, cfloat rhs)
    {
        return detail::abs(lhs) < detail::abs(rhs) ? lhs :  rhs;
    }

    template<>
    __DH__ cdouble min<cdouble>(cdouble lhs, cdouble rhs)
    {
        return detail::abs(lhs) < detail::abs(rhs) ? lhs :  rhs;
    }

    template<typename T>
    static __DH__ T constant(double val)
    {
        return (T)(val);
    }

    template<>
    __DH__ cfloat  constant<cfloat >(double val)
    {
        cfloat  cval = {(float)val, 0};
        return cval;
    }

    template<>
    __DH__ cdouble constant<cdouble >(double val)
    {
        cdouble  cval = {val, 0};
        return cval;
    }
};

template<typename Ti, typename To, af_op_t op>
struct reduce_op
{
    __DH__ To init()
    {
        assert("Operation not supported" && 1 != 1);
        return ops::constant<To>(0);
    }

    __DH__ To calc(To out, Ti in)
    {
        assert("Operation not supported" && 1 != 1);
        return out;
    }
};

template<typename Ti, typename To>
struct reduce_op<Ti, To, af_add_t>
{
    __DH__ To init()
    {
        return ops::constant<To>(0);
    }

    __DH__ To calc(Ti in, To out)
    {
        return in + out;
    }
};

template<typename Ti, typename To>
struct reduce_op<Ti, To, af_or_t>
{
    __DH__ To init()
    {
        return ops::constant<To>(0);
    }

    __DH__ To calc(Ti in, To out)
    {
        return out || (in != ops::constant<Ti>(0));
    }
};

template<typename Ti, typename To>
struct reduce_op<Ti, To, af_and_t>
{
    __DH__ To init()
    {
        return ops::constant<To>(1);
    }

    __DH__ To calc(Ti in, To out)
    {
        return out && (in != ops::constant<Ti>(0));
    }
};

template<typename Ti, typename To>
struct reduce_op<Ti, To, af_notzero_t>
{
    __DH__ To init()
    {
        return ops::constant<To>(0);
    }

    __DH__ To calc(Ti in, To out)
    {
        return out + (in != ops::constant<Ti>(0));
    }
};

template<typename Ti, typename To>
struct reduce_op<Ti, To, af_min_t>
{
    __DH__ To init()
    {
        return std::numeric_limits<To>::max();
    }

    __DH__ To calc(Ti in, To out)
    {
        return ops::min(out, in);
    }
};

#define SPEICALIZE_COMPLEX_MIN(T, Tr)                                   \
    template<>                                                          \
    struct reduce_op<T, T, af_min_t>                                    \
    {                                                                   \
        __DH__ T init()                                                 \
        {                                                               \
            return ops::constant<T>(                                    \
                std::numeric_limits<Tr>::max()                          \
                );                                                      \
        }                                                               \
                                                                        \
        __DH__ T calc(T in, T out)                                      \
        {                                                               \
            return ops::min(out, in);                                   \
        }                                                               \
    };                                                                  \

SPEICALIZE_COMPLEX_MIN(cfloat, float)
SPEICALIZE_COMPLEX_MIN(cdouble, double)

#undef SPEICALIZE_COMPLEX_MIN

template<typename Ti, typename To>
struct reduce_op<Ti, To, af_max_t>
{
    __DH__ To init()
    {
        return std::numeric_limits<To>::min();
    }

    __DH__ To calc(Ti in, To out)
    {
        return ops::max(out, in);
    }
};


#define SPEICALIZE_FLOATING_MAX(T)                  \
    template<>                                      \
    struct reduce_op<T, T, af_max_t>                \
    {                                               \
        __DH__ T init()                             \
        {                                           \
            return -std::numeric_limits<T>::max();  \
        }                                           \
                                                    \
        __DH__ T calc(T in, T out)                  \
        {                                           \
            return ops::max(out, in);               \
        }                                           \
    };                                              \

SPEICALIZE_FLOATING_MAX(float)
SPEICALIZE_FLOATING_MAX(double)

#undef SPEICALIZE_FLOATING_MAX
