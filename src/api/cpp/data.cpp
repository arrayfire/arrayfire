/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/complex.h>
#include <af/arith.h>
#include <af/data.h>
#include <af/traits.hpp>
#include <af/gfor.h>
#include "error.hpp"
#include <af/defines.h>

namespace af
{

    template<typename T>
    array
    constant(T val, const dim4 &dims, const af::dtype type)
    {
        af_array res;
        if (type != s64 && type != u64) {
            AF_THROW(af_constant(&res, (double)val,
                                 dims.ndims(), dims.get(), type));
        }
        else if (type == s64) {
                AF_THROW(af_constant_long (&res, ( intl)val,
                                           dims.ndims(),
                                           dims.get()));
        } else {
            AF_THROW(af_constant_ulong(&res, (uintl)val,
                                       dims.ndims(),
                                       dims.get()));
        }
        return array(res);
    }

    template<>
    AFAPI array constant(cfloat val, const dim4 &dims, const af::dtype type)
    {
        if (type != c32 && type != c64) {
            return constant(real(val), dims, type);
        }
        af_array res;
        AF_THROW(af_constant_complex(&res,
                                     real(val),
                                     imag(val),
                                     dims.ndims(),
                                     dims.get(), type));
        return array(res);
    }

    template<>
    AFAPI array constant(cdouble val, const dim4 &dims, const af::dtype type)
    {
        if (type != c32 && type != c64) {
            return constant(real(val), dims, type);
        }
        af_array res;
        AF_THROW(af_constant_complex(&res,
                                     real(val),
                                     imag(val),
                                     dims.ndims(),
                                     dims.get(), type));
        return array(res);
    }
    template<typename T>
    array constant(T val, const dim_t d0, const af::dtype ty)
    {
        return constant(val, dim4(d0), ty);
    }

    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const af::dtype ty)
    {
        return constant(val, dim4(d0, d1), ty);
    }

    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const dim_t d2, const af::dtype ty)
    {
        return constant(val, dim4(d0, d1, d2), ty);
    }

    template<typename T>
    array constant(T val, const dim_t d0, const dim_t d1, const dim_t d2, const dim_t d3, const af::dtype ty)
    {
        return constant(val, dim4(d0, d1, d2, d3), ty);
    }

#define CONSTANT(TYPE)                                                              \
    template AFAPI array constant<TYPE>(TYPE val, const dim4 &dims, const af::dtype ty);  \
    template AFAPI array constant<TYPE>(TYPE val, const dim_t d0, const af::dtype ty);    \
    template AFAPI array constant<TYPE>(TYPE val, const dim_t d0,                         \
                                            const dim_t d1, const af::dtype ty);    \
    template AFAPI array constant<TYPE>(TYPE val, const dim_t d0,                         \
                                            const dim_t d1,                         \
                                            const dim_t d2, const af::dtype ty);    \
    template AFAPI array constant<TYPE>(TYPE val, const dim_t d0,                         \
                                            const dim_t d1,                         \
                                            const dim_t d2,                         \
                                            const dim_t d3, const af::dtype ty);
    CONSTANT(double);
    CONSTANT(float);
    CONSTANT(int);
    CONSTANT(unsigned);
    CONSTANT(char);
    CONSTANT(unsigned char);
    CONSTANT(cfloat);
    CONSTANT(cdouble);
    CONSTANT(long);
    CONSTANT(unsigned long);
    CONSTANT(long long);
    CONSTANT(unsigned long long);
    CONSTANT(bool);
    CONSTANT(short);
    CONSTANT(unsigned short);

#undef CONSTANT

    array range(const dim4 &dims, const int seq_dim, const af::dtype ty)
    {
        af_array out;
        AF_THROW(af_range(&out, dims.ndims(), dims.get(), seq_dim, ty));
        return array(out);
    }

    array range(const dim_t d0, const dim_t d1, const dim_t d2,
               const dim_t d3, const int seq_dim, const af::dtype ty)
    {
        return range(dim4(d0, d1, d2, d3), seq_dim, ty);
    }

    array iota(const dim4 &dims, const dim4 &tile_dims, const af::dtype ty)
    {
        af_array out;
        AF_THROW(af_iota(&out, dims.ndims(), dims.get(), tile_dims.ndims(), tile_dims.get(), ty));
        return array(out);
    }

    array identity(const dim4 &dims, const af::dtype type)
    {
        af_array res;
        AF_THROW(af_identity(&res, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array identity(const dim_t d0, const af::dtype ty)
    {
        return identity(dim4(d0), ty);
    }

    array identity(const dim_t d0,
                const dim_t d1, const af::dtype ty)
    {
        return identity(dim4(d0, d1), ty);
    }

    array identity(const dim_t d0,
                const dim_t d1, const dim_t d2, const af::dtype ty)
    {
        return identity(dim4(d0, d1, d2), ty);
    }

    array identity(const dim_t d0,
                const dim_t d1, const dim_t d2,
                const dim_t d3, const af::dtype ty)
    {
        return identity(dim4(d0, d1, d2, d3), ty);
    }

    array diag(const array &in, const int num, const bool extract)
    {
        af_array res;
        if (extract) {
            AF_THROW(af_diag_extract(&res, in.get(), num));
        } else {
            AF_THROW(af_diag_create(&res, in.get(), num));
        }

        return array(res);
    }

    array moddims(const array& in, const unsigned ndims, const dim_t * const dims)
    {
        af_array out = 0;
        AF_THROW(af_moddims(&out, in.get(), ndims, dims));
        return array(out);
    }

    array moddims(const array& in, const dim4& dims)
    {
        return af::moddims(in, dims.ndims(), dims.get());
    }

    array moddims(const array& in, const dim_t d0, const dim_t d1, const dim_t d2, const dim_t d3)
    {
        dim_t dims[4] = {d0, d1, d2, d3};
        return af::moddims(in, 4, dims);
    }

    array flat(const array& in)
    {
        af_array out = 0;
        AF_THROW(af_flat(&out, in.get()));
        return array(out);
    }

    array join(const int dim, const array& first, const array& second)
    {
        af_array out = 0;
        AF_THROW(af_join(&out, dim, first.get(), second.get()));
        return array(out);
    }

    array join(const int dim, const array& first, const array& second, const array &third)
    {
        af_array out = 0;
        af_array inputs[3] = {first.get(), second.get(), third.get()};
        AF_THROW(af_join_many(&out, dim, 3, inputs));
        return array(out);
    }

    array join(const int dim, const array& first, const array& second, const array &third, const array &fourth)
    {
        af_array out = 0;
        af_array inputs[4] = {first.get(), second.get(), third.get(), fourth.get()};
        AF_THROW(af_join_many(&out, dim, 4, inputs));
        return array(out);
    }

    array tile(const array& in, const unsigned x, const unsigned y, const unsigned z, const unsigned w)
    {
        af_array out = 0;
        AF_THROW(af_tile(&out, in.get(), x, y, z, w));
        return array(out);
    }

    array tile(const array& in, const af::dim4 &dims)
    {
        af_array out = 0;
        AF_THROW(af_tile(&out, in.get(), dims[0], dims[1], dims[2], dims[3]));
        return array(out);
    }

    array reorder(const array& in, const unsigned x, const unsigned y, const unsigned z, const unsigned w)
    {
        af_array out = 0;
        AF_THROW(af_reorder(&out, in.get(), x, y, z, w));
        return array(out);
    }

    array shift(const array& in, const int x, const int y, const int z, const int w)
    {
        af_array out = 0;
        AF_THROW(af_shift(&out, in.get(), x, y, z, w));
        return array(out);
    }

    array flip(const array &in, const unsigned dim)
    {
        af_array out = 0;
        AF_THROW(af_flip(&out, in.get(), dim));
        return array(out);
    }

    array lower(const array &in, bool is_unit_diag)
    {
        af_array res;
        AF_THROW(af_lower(&res, in.get(), is_unit_diag));
        return array(res);
    }

    array upper(const array &in, bool is_unit_diag)
    {
        af_array res;
        AF_THROW(af_upper(&res, in.get(), is_unit_diag));
        return array(res);
    }

    array select(const array &cond, const array &a, const array &b)
    {
        af_array res;
        AF_THROW(af_select(&res, cond.get(), a.get(), b.get()));
        return array(res);
    }

    array select(const array &cond, const array &a, const double &b)
    {
        af_array res;
        AF_THROW(af_select_scalar_r(&res, cond.get(), a.get(), b));
        return array(res);
    }

    array select(const array &cond, const double &a, const array &b)
    {
        af_array res;
        AF_THROW(af_select_scalar_l(&res, cond.get(), a, b.get()));
        return array(res);
    }

    void replace(array &a, const array &cond, const array &b)
    {
        AF_THROW(af_replace(a.get(), cond.get(), b.get()));
    }

    void replace(array &a, const array &cond, const double &b)
    {
        AF_THROW(af_replace_scalar(a.get(), cond.get(), b));
    }
}
