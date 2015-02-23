#include <af/array.h>
#include <af/arith.h>
#include <af/data.h>
#include <af/traits.hpp>
#include <af/index.h>
#include "error.hpp"

namespace af
{

#define CONSTANT(TYPE)                                                  \
    array constant(TYPE val, const dim_type d0, af::dtype ty)           \
    {                                                                   \
        return constant(val, dim4(d0), ty);                             \
    }                                                                   \
                                                                        \
    array constant(TYPE val, const dim_type d0,                         \
                   const dim_type d1, af::dtype ty)                     \
    {                                                                   \
        return constant(val, dim4(d0, d1), ty);                         \
    }                                                                   \
                                                                        \
    array constant(TYPE val, const dim_type d0,                         \
                   const dim_type d1, const dim_type d2, af::dtype ty)  \
    {                                                                   \
        return constant(val, dim4(d0, d1, d2), ty);                     \
    }                                                                   \
                                                                        \
    array constant(TYPE val, const dim_type d0,                         \
                   const dim_type d1, const dim_type d2,                \
                   const dim_type d3, af::dtype ty)                     \
    {                                                                   \
        return constant(val, dim4(d0, d1, d2, d3), ty);                 \
    }                                                                   \

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

#undef CONSTANT

#define CONSTANT_DOUBLE(TYPE)                                   \
    array constant(TYPE val, const dim4 &dims, af::dtype type)  \
    {                                                           \
        af_array res;                                           \
        AF_THROW(af_constant(&res, val,                         \
                             dims.ndims(), dims.get(), type));  \
        return array(res);                                      \
    }                                                           \

    CONSTANT_DOUBLE(double)
    CONSTANT_DOUBLE(float)
    CONSTANT_DOUBLE(int)
    CONSTANT_DOUBLE(unsigned)
    CONSTANT_DOUBLE(char)
    CONSTANT_DOUBLE(unsigned char)

#undef CONSTANT_DOUBLE

#define CONSTANT_LONG(TYPE, DTYPE)                              \
    array constant(TYPE val, const dim4 &dims, af::dtype type)  \
    {                                                           \
        if (type != s64 && type != u64) {                       \
            return constant((double)val, dims, type);           \
        }                                                       \
        af_array res;                                           \
        if (DTYPE == s64) {                                     \
            AF_THROW(af_constant_long (&res, ( intl)val,        \
                                       dims.ndims(),            \
                                       dims.get()));            \
        } else {                                                \
            AF_THROW(af_constant_ulong(&res, (uintl)val,        \
                                       dims.ndims(),            \
                                       dims.get()));            \
        }                                                       \
        return array(res);                                      \
    }                                                           \

    CONSTANT_LONG(long, s64)
    CONSTANT_LONG(long long, s64)
    CONSTANT_LONG(unsigned long, u64)
    CONSTANT_LONG(unsigned long long, u64)

#undef CONSTANT_LONG

#define CONSTANT_COMPLEX(TYPE)                                  \
    array constant(TYPE val, const dim4 &dims, af::dtype type)  \
    {                                                           \
        if (type != c32 && type != c64) {                       \
            return constant(real(val), dims, type);             \
        }                                                       \
        af_array res;                                           \
        AF_THROW(af_constant_complex(&res,                      \
                                     real(val),                 \
                                     imag(val),                 \
                                     dims.ndims(),              \
                                     dims.get(), type));        \
        return array(res);                                      \
    }                                                           \

    CONSTANT_COMPLEX(cdouble)
    CONSTANT_COMPLEX(cfloat)

#undef CONSTANT_COMPLEX

    array randu(const dim4 &dims, af::dtype type)
    {
        af_array res;
        AF_THROW(af_randu(&res, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array randu(const dim_type d0, af::dtype ty)
    {
        return randu(dim4(d0), ty);
    }

    array randu(const dim_type d0,
                const dim_type d1, af::dtype ty)
    {
        return randu(dim4(d0, d1), ty);
    }

    array randu(const dim_type d0,
                const dim_type d1, const dim_type d2, af::dtype ty)
    {
        return randu(dim4(d0, d1, d2), ty);
    }

    array randu(const dim_type d0,
                const dim_type d1, const dim_type d2,
                const dim_type d3, af::dtype ty)
    {
        return randu(dim4(d0, d1, d2, d3), ty);
    }

    array randn(const dim4 &dims, af::dtype type)
    {
        af_array res;
        AF_THROW(af_randn(&res, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array randn(const dim_type d0, af::dtype ty)
    {
        return randn(dim4(d0), ty);
    }

    array randn(const dim_type d0,
                const dim_type d1, af::dtype ty)
    {
        return randn(dim4(d0, d1), ty);
    }

    array randn(const dim_type d0,
                const dim_type d1, const dim_type d2, af::dtype ty)
    {
        return randn(dim4(d0, d1, d2), ty);
    }

    array randn(const dim_type d0,
                const dim_type d1, const dim_type d2,
                const dim_type d3, af::dtype ty)
    {
        return randn(dim4(d0, d1, d2, d3), ty);
    }

    array iota(const dim4 &dims, const int rep, af::dtype ty)
    {
        af_array out;
        AF_THROW(af_iota(&out, dims.ndims(), dims.get(), rep, ty));
        return array(out);
    }

    array iota(const dim_type d0, const dim_type d1, const dim_type d2,
               const dim_type d3, const int rep, af::dtype ty)
    {
        return iota(dim4(d0, d1, d2, d3), rep, ty);
    }

    array identity(const dim4 &dims, af::dtype type)
    {
        af_array res;
        AF_THROW(af_identity(&res, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array identity(const dim_type d0, af::dtype ty)
    {
        return identity(dim4(d0), ty);
    }

    array identity(const dim_type d0,
                const dim_type d1, af::dtype ty)
    {
        return identity(dim4(d0, d1), ty);
    }

    array identity(const dim_type d0,
                const dim_type d1, const dim_type d2, af::dtype ty)
    {
        return identity(dim4(d0, d1, d2), ty);
    }

    array identity(const dim_type d0,
                const dim_type d1, const dim_type d2,
                const dim_type d3, af::dtype ty)
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
}
