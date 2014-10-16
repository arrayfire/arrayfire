#pragma once
#include <backend.hpp>
#include <math.hpp>

#ifndef __DH__
#define __DH__
#endif

#include "optypes.hpp"

using namespace detail;

template<typename T, af_op_t op>
struct Binary
{
    __DH__ T init()
    {
        return detail::scalar<T>(0);
    }

    __DH__ T operator() (T lhs, T rhs)
    {
        return lhs + rhs;
    }
};

template<typename T>
struct Binary<T, af_add_t>
{
    __DH__ T init()
    {
        return detail::scalar<T>(0);
    }

    __DH__ T operator() (T lhs, T rhs)
    {
        return lhs + rhs;
    }
};

template<typename T>
struct Binary<T, af_or_t>
{
    __DH__ T init()
    {
        return detail::scalar<T>(0);
    }

    __DH__ T operator() (T lhs, T rhs)
    {
        return lhs || rhs;
    }
};

template<typename T>
struct Binary<T, af_and_t>
{
    __DH__ T init()
    {
        return detail::scalar<T>(1);
    }

    __DH__ T operator() (T lhs, T rhs)
    {
        return lhs && rhs;
    }
};

template<typename T>
struct Binary<T, af_notzero_t>
{
    __DH__ T init()
    {
        return detail::scalar<T>(0);
    }

    __DH__ T operator() (T lhs, T rhs)
    {
        return lhs + rhs;
    }
};

template<typename T>
struct Binary<T, af_min_t>
{
    __DH__ T init()
    {
        return detail::limit_max<T>();
    }

    __DH__ T operator() (T lhs, T rhs)
    {
        return detail::min(lhs, rhs);
    }
};

#define SPECIALIZE_COMPLEX_MIN(T, Tr)           \
    template<>                                  \
    struct Binary<T, af_min_t>                  \
    {                                           \
        __DH__ T init()                         \
        {                                       \
            return detail::scalar<T>(           \
                detail::limit_max<Tr>()         \
                );                              \
        }                                       \
                                                \
        __DH__ T operator() (T lhs, T rhs)      \
        {                                       \
            return detail::min(lhs, rhs);       \
        }                                       \
    };                                          \

SPECIALIZE_COMPLEX_MIN(cfloat, float)
SPECIALIZE_COMPLEX_MIN(cdouble, double)

#undef SPECIALIZE_COMPLEX_MIN

template<typename T>
struct Binary<T, af_max_t>
{
    __DH__ T init()
    {
        return detail::limit_min<T>();
    }

    __DH__ T operator() (T lhs, T rhs)
    {
        return detail::max(lhs, rhs);
    }
};


#define SPECIALIZE_FLOATING_MAX(T, Tr)          \
    template<>                                  \
    struct Binary<T, af_max_t>                  \
    {                                           \
        __DH__ T init()                         \
        {                                       \
            return detail::scalar<T>(           \
                -detail::limit_max<Tr>()        \
                );                              \
        }                                       \
                                                \
        __DH__ T operator() (T lhs, T rhs)      \
        {                                       \
            return detail::max(lhs, rhs);       \
        }                                       \
    };                                          \

SPECIALIZE_FLOATING_MAX(float, float)
SPECIALIZE_FLOATING_MAX(double, double)
SPECIALIZE_FLOATING_MAX(cfloat, float)
SPECIALIZE_FLOATING_MAX(cdouble, double)

#undef SPECIALIZE_FLOATING_MAX

template<typename Ti, typename To, af_op_t op>
struct Transform
{
    __DH__ To operator ()(Ti in)
    {
        return (To)(in);
    }
};

template<typename Ti, typename To>
struct Transform<Ti, To, af_or_t>
{
    __DH__ To operator ()(Ti in)
    {
        return (in != detail::scalar<Ti>(0));
    }
};

template<typename Ti, typename To>
struct Transform<Ti, To, af_and_t>
{
    __DH__ To operator ()(Ti in)
    {
        return (in != detail::scalar<Ti>(0));
    }
};

template<typename Ti, typename To>
struct Transform<Ti, To, af_notzero_t>
{
    __DH__ To operator ()(Ti in)
    {
        return (in != detail::scalar<Ti>(0));
    }
};
