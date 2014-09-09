#pragma once
#include <backend.hpp>
#include <math.hpp>

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

#define SPEICALIZE_COMPLEX_MIN(T, Tr)           \
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

SPEICALIZE_COMPLEX_MIN(cfloat, float)
SPEICALIZE_COMPLEX_MIN(cdouble, double)

#undef SPEICALIZE_COMPLEX_MIN

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


#define SPEICALIZE_FLOATING_MAX(T)              \
    template<>                                  \
    struct Binary<T, af_max_t>                  \
    {                                           \
        __DH__ T init()                         \
        {                                       \
            return -detail::limit_max<T>();     \
        }                                       \
                                                \
        __DH__ T operator() (T lhs, T rhs)      \
        {                                       \
            return detail::max(lhs, rhs);       \
        }                                       \
    };                                          \

SPEICALIZE_FLOATING_MAX(float)
SPEICALIZE_FLOATING_MAX(double)

#undef SPEICALIZE_FLOATING_MAX

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
