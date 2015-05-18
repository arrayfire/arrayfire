/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "magma_common.h"

template<typename T> T magma_one() { return (T)1.0; }
template<typename T> T magma_neg_one() { return (T)-1.0; }
template<typename T> T magma_zero() { return (T)0; }

#define INSTANTIATE_REAL(func, T)          \
    template T func<T>();

INSTANTIATE_REAL(magma_one    , float )
INSTANTIATE_REAL(magma_neg_one, float )
INSTANTIATE_REAL(magma_zero   , float )
INSTANTIATE_REAL(magma_one    , double)
INSTANTIATE_REAL(magma_neg_one, double)
INSTANTIATE_REAL(magma_zero   , double)

#define INSTANTIATE_CPLX(func, T, val)          \
    template<> T func<T>()                      \
    {                                           \
        T res;                                  \
        res.s[0] = val;                         \
        res.s[1] = 0;                           \
        return res;                             \
    }                                           \

INSTANTIATE_CPLX(magma_one    , magmaFloatComplex ,  1.0)
INSTANTIATE_CPLX(magma_neg_one, magmaFloatComplex , -1.0)
INSTANTIATE_CPLX(magma_zero   , magmaFloatComplex ,  0.0)
INSTANTIATE_CPLX(magma_one    , magmaDoubleComplex,  1.0)
INSTANTIATE_CPLX(magma_neg_one, magmaDoubleComplex, -1.0)
INSTANTIATE_CPLX(magma_zero   , magmaDoubleComplex,  0.0)

template<typename T> T magma_scalar(double val) { return (T)val; }
template float magma_scalar<float>(double val);
template double magma_scalar<double>(double val);

#define INSTANTIATE_CPLX_SCALAR(T)              \
    template<> T magma_scalar<T>(double val)    \
    {                                           \
        T res;                                  \
        res.s[0] = val;                         \
        res.s[1] = 0;                           \
        return res;                             \
    }                                           \

INSTANTIATE_CPLX_SCALAR(magmaFloatComplex);
INSTANTIATE_CPLX_SCALAR(magmaDoubleComplex);

template<typename T> bool magma_is_real() { return true; }
template bool magma_is_real<float>();
template bool magma_is_real<double>();
template<> bool magma_is_real<magmaFloatComplex>() { return false; }
template<> bool magma_is_real<magmaDoubleComplex>() { return false; }

template<typename T>
magma_int_t magma_get_getrf_nb(magma_int_t m )
{
    if      (m <= 3200) return 128;
    else if (m <  9000) return 256;
    else                return 320;
}

template magma_int_t magma_get_getrf_nb<float>(magma_int_t m);

template<>
magma_int_t magma_get_getrf_nb<double>( magma_int_t m )
{
    if      (m <= 2048) return 64;
    else if (m <  7200) return 192;
    else                return 256;
}

template<>
magma_int_t magma_get_getrf_nb<magmaFloatComplex>( magma_int_t m )
{
    if      (m <= 2048) return 64;
    else                return 128;
}

template<>
magma_int_t magma_get_getrf_nb<magmaDoubleComplex>( magma_int_t m )
{
    if      (m <= 3072) return 32;
    else if (m <= 9024) return 64;
    else                return 128;
}

template<typename T>
magma_int_t magma_get_potrf_nb(magma_int_t m )
{
    if      (m <= 1024) return 128;
    else                return 320;
}

template magma_int_t magma_get_potrf_nb<float>(magma_int_t m);

template<>
magma_int_t magma_get_potrf_nb<double>(magma_int_t m)
{
    if      (m <= 4256) return 128;
    else                return 256;
}

template<>
magma_int_t magma_get_potrf_nb<magmaFloatComplex>(magma_int_t m)
{
    return 128;
}

template<>
magma_int_t magma_get_potrf_nb<magmaDoubleComplex>(magma_int_t m)
{
    return  64;
}

template<typename T>
magma_int_t magma_get_geqrf_nb(magma_int_t m )
{
    return 128;
}

template magma_int_t magma_get_geqrf_nb<float>(magma_int_t m);

template<>
magma_int_t magma_get_geqrf_nb<double>( magma_int_t m )
{
    if      (m <= 2048) return 64;
    return 128;
}

template<>
magma_int_t magma_get_geqrf_nb<magmaFloatComplex>( magma_int_t m )
{
    if      (m <= 2048) return 32;
    else if (m <= 4032) return 64;
    else                return 128;
}

template<>
magma_int_t magma_get_geqrf_nb<magmaDoubleComplex>( magma_int_t m )
{
    if      (m <= 2048) return 32;
    else if (m <= 4032) return 64;
    else                return 128;
}
