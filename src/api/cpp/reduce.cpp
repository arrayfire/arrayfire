/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/algorithm.h>
#include "error.hpp"

namespace af
{
    array sum(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_sum(&out, in.get(), dim));
        return array(out);
    }

    array product(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_product(&out, in.get(), dim));
        return array(out);
    }

    array min(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_min(&out, in.get(), dim));
        return array(out);
    }

    array max(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_max(&out, in.get(), dim));
        return array(out);
    }

    array alltrue(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_alltrue(&out, in.get(), dim));
        return array(out);
    }

    array anytrue(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_anytrue(&out, in.get(), dim));
        return array(out);
    }

    array count(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_count(&out, in.get(), dim));
        return array(out);
    }

    void min(array &val, array &idx, const array &in, const int dim)
    {
        af_array out = 0;
        af_array loc = 0;
        AF_THROW(af_imin(&out, &loc, in.get(), dim));
        val = array(out);
        idx = array(loc);
    }

    void max(array &val, array &idx, const array &in, const int dim)
    {
        af_array out = 0;
        af_array loc = 0;
        AF_THROW(af_imax(&out, &loc, in.get(), dim));
        val = array(out);
        idx = array(loc);
    }

#define INSTANTIATE_REAL(fn, T)                             \
    template<> AFAPI                                        \
    T fn(const array &in)                                   \
    {                                                       \
        double rval, ival;                                  \
        AF_THROW(af_##fn##_all(&rval, &ival, in.get()));    \
        return (T)(rval);                                   \
    }                                                       \


#define INSTANTIATE_CPLX(fn, T, Tr)                         \
    template<> AFAPI                                        \
    T fn(const array &in)                                   \
    {                                                       \
        double rval, ival;                                  \
        AF_THROW(af_##fn##_all(&rval, &ival, in.get()));    \
        T out = {(Tr)rval, (Tr)ival};                       \
        return out;                                         \
    }                                                       \

#define INSTANTIATE(fn)                         \
    INSTANTIATE_REAL(fn, float)                 \
    INSTANTIATE_REAL(fn, double)                \
    INSTANTIATE_REAL(fn, int)                   \
    INSTANTIATE_REAL(fn, unsigned)              \
    INSTANTIATE_REAL(fn, char)                  \
    INSTANTIATE_REAL(fn, unsigned char)         \
    INSTANTIATE_CPLX(fn, af_cfloat, float)      \
    INSTANTIATE_CPLX(fn, af_cdouble, double)    \

    INSTANTIATE(sum)
    INSTANTIATE(product)
    INSTANTIATE(min)
    INSTANTIATE(max)
    INSTANTIATE(alltrue)
    INSTANTIATE(anytrue)
    INSTANTIATE(count)

#undef INSTANTIATE
#undef INSTANTIATE_REAL
#undef INSTANTIATE_CPLX

#define INSTANTIATE_REAL(fn, T)                                 \
    template<> AFAPI                                            \
    void fn(T *val, unsigned *idx, const array &in)             \
    {                                                           \
        double rval, ival;                                      \
        AF_THROW(af_i##fn##_all(&rval, &ival, idx, in.get()));  \
        *val = (T)(rval);                                       \
    }                                                           \


#define INSTANTIATE_CPLX(fn, T, Tr)                             \
    template<> AFAPI                                            \
    void fn(T *val, unsigned *idx, const array &in)             \
    {                                                           \
        double rval, ival;                                      \
        AF_THROW(af_i##fn##_all(&rval, &ival, idx, in.get()));  \
        *val = {(Tr)rval, (Tr)ival};                            \
    }                                                           \

#define INSTANTIATE(fn)                         \
    INSTANTIATE_REAL(fn, float)                 \
    INSTANTIATE_REAL(fn, double)                \
    INSTANTIATE_REAL(fn, int)                   \
    INSTANTIATE_REAL(fn, unsigned)              \
    INSTANTIATE_REAL(fn, char)                  \
    INSTANTIATE_REAL(fn, unsigned char)         \
    INSTANTIATE_CPLX(fn, af_cfloat, float)      \
    INSTANTIATE_CPLX(fn, af_cdouble, double)    \

    INSTANTIATE(min)
    INSTANTIATE(max)
}
