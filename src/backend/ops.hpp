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

template<typename T, af_op_t op>
struct reduce_op
{
    __DH__ T init()
    {
        return ops::constant<T>(0);
    }

    __DH__ T calc(T lhs, T rhs)
    {
        return lhs + rhs;
    }
};

template<typename T>
struct reduce_op<T, af_add_t>
{
    __DH__ T init()
    {
        return ops::constant<T>(0);
    }

    __DH__ T calc(T lhs, T rhs)
    {
        return lhs + rhs;
    }
};

template<typename T>
struct reduce_op<T, af_or_t>
{
    __DH__ T init()
    {
        return ops::constant<T>(0);
    }

    __DH__ T calc(T lhs, T rhs)
    {
        return lhs || rhs;
    }
};

template<typename T>
struct reduce_op<T, af_and_t>
{
    __DH__ T init()
    {
        return ops::constant<T>(1);
    }

    __DH__ T calc(T lhs, T rhs)
    {
        return lhs && rhs;
    }
};

template<typename T>
struct reduce_op<T, af_notzero_t>
{
    __DH__ T init()
    {
        return ops::constant<T>(0);
    }

    __DH__ T calc(T lhs, T rhs)
    {
        return lhs + rhs;
    }
};

template<typename T>
struct reduce_op<T, af_min_t>
{
    __DH__ T init()
    {
        return std::numeric_limits<T>::max();
    }

    __DH__ T calc(T lhs, T rhs)
    {
        return ops::min(lhs, rhs);
    }
};

#define SPEICALIZE_COMPLEX_MIN(T, Tr)           \
    template<>                                  \
    struct reduce_op<T, af_min_t>               \
    {                                           \
        __DH__ T init()                         \
        {                                       \
            return ops::constant<T>(            \
                std::numeric_limits<Tr>::max()  \
                );                              \
        }                                       \
                                                \
        __DH__ T calc(T lhs, T rhs)             \
        {                                       \
            return ops::min(lhs, rhs);          \
        }                                       \
    };                                          \

SPEICALIZE_COMPLEX_MIN(cfloat, float)
SPEICALIZE_COMPLEX_MIN(cdouble, double)

#undef SPEICALIZE_COMPLEX_MIN

template<typename T>
struct reduce_op<T, af_max_t>
{
    __DH__ T init()
    {
        return std::numeric_limits<T>::min();
    }

    __DH__ T calc(T lhs, T rhs)
    {
        return ops::max(lhs, rhs);
    }
};


#define SPEICALIZE_FLOATING_MAX(T)                  \
    template<>                                      \
    struct reduce_op<T, af_max_t>                   \
    {                                               \
        __DH__ T init()                             \
        {                                           \
            return -std::numeric_limits<T>::max();  \
        }                                           \
                                                    \
        __DH__ T calc(T lhs, T rhs)                 \
        {                                           \
            return ops::max(lhs, rhs);              \
        }                                           \
    };                                              \

SPEICALIZE_FLOATING_MAX(float)
SPEICALIZE_FLOATING_MAX(double)

#undef SPEICALIZE_FLOATING_MAX

template<typename Ti, typename To, af_op_t op>
struct transform_op
{
    __DH__ To operator ()(Ti in)
    {
        return (To)(in);
    }
};

template<typename Ti, typename To>
struct transform_op<Ti, To, af_or_t>
{
    __DH__ To operator ()(Ti in)
    {
        return (in != ops::constant<Ti>(0));
    }
};

template<typename Ti, typename To>
struct transform_op<Ti, To, af_and_t>
{
    __DH__ To operator ()(Ti in)
    {
        return (in != ops::constant<Ti>(0));
    }
};

template<typename Ti, typename To>
struct transform_op<Ti, To, af_notzero_t>
{
    __DH__ To operator ()(Ti in)
    {
        return (in != ops::constant<Ti>(0));
    }
};
