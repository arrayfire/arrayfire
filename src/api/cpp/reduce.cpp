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
#include "common.hpp"

namespace af
{
    array sum(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_sum(&out, in.get(), getFNSD(dim, in.dims())));
        return array(out);
    }

    array product(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_product(&out, in.get(), getFNSD(dim, in.dims())));
        return array(out);
    }

    array min(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_min(&out, in.get(), getFNSD(dim, in.dims())));
        return array(out);
    }

    array max(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_max(&out, in.get(), getFNSD(dim, in.dims())));
        return array(out);
    }

    array allTrue(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_all_true(&out, in.get(), getFNSD(dim, in.dims())));
        return array(out);
    }

    array anyTrue(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_any_true(&out, in.get(), getFNSD(dim, in.dims())));
        return array(out);
    }

    array count(const array &in, const int dim)
    {
        af_array out = 0;
        AF_THROW(af_count(&out, in.get(), getFNSD(dim, in.dims())));
        return array(out);
    }

    void min(array &val, array &idx, const array &in, const int dim)
    {
        af_array out = 0;
        af_array loc = 0;
        AF_THROW(af_imin(&out, &loc, in.get(), getFNSD(dim, in.dims())));
        val = array(out);
        idx = array(loc);
    }

    void max(array &val, array &idx, const array &in, const int dim)
    {
        af_array out = 0;
        af_array loc = 0;
        AF_THROW(af_imax(&out, &loc, in.get(), getFNSD(dim, in.dims())));
        val = array(out);
        idx = array(loc);
    }

#define INSTANTIATE_REAL(fnC, fnCPP, T)                             \
    template<> AFAPI                                        \
    T fnCPP(const array &in)                                   \
    {                                                       \
        double rval, ival;                                  \
        AF_THROW(af_##fnC##_all(&rval, &ival, in.get()));    \
        return (T)(rval);                                   \
    }                                                       \


#define INSTANTIATE_CPLX(fnC, fnCPP, T, Tr)                         \
    template<> AFAPI                                        \
    T fnCPP(const array &in)                                   \
    {                                                       \
        double rval, ival;                                  \
        AF_THROW(af_##fnC##_all(&rval, &ival, in.get()));    \
        T out((Tr)rval, (Tr)ival);                       \
        return out;                                         \
    }                                                       \

#define INSTANTIATE(fnC, fnCPP)                         \
    INSTANTIATE_REAL(fnC, fnCPP, float)                 \
    INSTANTIATE_REAL(fnC, fnCPP, double)                \
    INSTANTIATE_REAL(fnC, fnCPP, int)                   \
    INSTANTIATE_REAL(fnC, fnCPP, unsigned)              \
    INSTANTIATE_REAL(fnC, fnCPP, long)                  \
    INSTANTIATE_REAL(fnC, fnCPP, unsigned long)         \
    INSTANTIATE_REAL(fnC, fnCPP, long long)             \
    INSTANTIATE_REAL(fnC, fnCPP, unsigned long long)    \
    INSTANTIATE_REAL(fnC, fnCPP, char)                  \
    INSTANTIATE_REAL(fnC, fnCPP, unsigned char)         \
    INSTANTIATE_CPLX(fnC, fnCPP, af_cfloat, float)      \
    INSTANTIATE_CPLX(fnC, fnCPP, af_cdouble, double)    \

    INSTANTIATE(sum, sum)
    INSTANTIATE(product, product)
    INSTANTIATE(min, min)
    INSTANTIATE(max, max)
    INSTANTIATE(all_true, allTrue)
    INSTANTIATE(any_true, anyTrue)
    INSTANTIATE(count, count)

    INSTANTIATE_REAL(all_true, allTrue, bool);
    INSTANTIATE_REAL(any_true, anyTrue, bool);

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
        *val = T((Tr)rval, (Tr)ival);                            \
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
