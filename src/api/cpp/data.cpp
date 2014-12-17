#include <af/array.h>
#include <af/arith.h>
#include <af/data.h>
#include <af/traits.hpp>
#include <af/index.h>
#include "error.hpp"

namespace af
{

    array constant(double val, const dim4 &dims, af_dtype type)
    {
        af_array res;
        AF_THROW(af_constant(&res, val, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array constant(cdouble val, const dim4 &dims)
    {
        af_array res;
        AF_THROW(af_constant_complex(&res, real(val), imag(val),
                                     dims.ndims(), dims.get(), c64));
        return array(res);
    }

    array constant(cfloat val, const dim4 &dims)
    {
        af_array res;
        AF_THROW(af_constant_complex(&res, real(val), imag(val),
                                     dims.ndims(), dims.get(), c32));
        return array(res);
    }

    array constant(double val, const dim_type d0, af_dtype ty)
    {
        return constant(val, dim4(d0), ty);
    }

    array constant(double val, const dim_type d0,
                         const dim_type d1, af_dtype ty)
    {
        return constant(val, dim4(d0, d1), ty);
    }

    array constant(double val, const dim_type d0,
                         const dim_type d1, const dim_type d2, af_dtype ty)
    {
        return constant(val, dim4(d0, d1, d2), ty);
    }

    array constant(double val, const dim_type d0,
                         const dim_type d1, const dim_type d2,
                         const dim_type d3, af_dtype ty)
    {
        return constant(val, dim4(d0, d1, d2, d3), ty);
    }


    array randu(const dim4 &dims, af_dtype type)
    {
        af_array res;
        AF_THROW(af_randu(&res, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array randu(const dim_type d0, af_dtype ty)
    {
        return randu(dim4(d0), ty);
    }

    array randu(const dim_type d0,
                const dim_type d1, af_dtype ty)
    {
        return randu(dim4(d0, d1), ty);
    }

    array randu(const dim_type d0,
                const dim_type d1, const dim_type d2, af_dtype ty)
    {
        return randu(dim4(d0, d1, d2), ty);
    }

    array randu(const dim_type d0,
                const dim_type d1, const dim_type d2,
                const dim_type d3, af_dtype ty)
    {
        return randu(dim4(d0, d1, d2, d3), ty);
    }

    array randn(const dim4 &dims, af_dtype type)
    {
        af_array res;
        AF_THROW(af_randn(&res, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array randn(const dim_type d0, af_dtype ty)
    {
        return randn(dim4(d0), ty);
    }

    array randn(const dim_type d0,
                const dim_type d1, af_dtype ty)
    {
        return randn(dim4(d0, d1), ty);
    }

    array randn(const dim_type d0,
                const dim_type d1, const dim_type d2, af_dtype ty)
    {
        return randn(dim4(d0, d1, d2), ty);
    }

    array randn(const dim_type d0,
                const dim_type d1, const dim_type d2,
                const dim_type d3, af_dtype ty)
    {
        return randn(dim4(d0, d1, d2, d3), ty);
    }

    array iota(const dim4 &dims, const unsigned rep, af_dtype ty)
    {
        af_array out;
        AF_THROW(af_iota(&out, dims.ndims(), dims.get(), rep, ty));
        return array(out);
    }

    array iota(const dim_type d0, const unsigned rep, af_dtype ty)
    {
        return iota(dim4(d0), rep, ty);
    }

    array iota(const dim_type d0, const dim_type d1,
               const unsigned rep, af_dtype ty)
    {
        return iota(dim4(d0, d1), rep, ty);
    }

    array iota(const dim_type d0, const dim_type d1, const dim_type d2,
               const unsigned rep, af_dtype ty)
    {
        return iota(dim4(d0, d1, d2), rep, ty);
    }

    array iota(const dim_type d0, const dim_type d1, const dim_type d2,
               const dim_type d3, const unsigned rep, af_dtype ty)
    {
        return iota(dim4(d0, d1, d2, d3), rep, ty);
    }

    array identity(const dim4 &dims, af_dtype type)
    {
        af_array res;
        AF_THROW(af_identity(&res, dims.ndims(), dims.get(), type));
        return array(res);
    }

    array identity(const dim_type d0, af_dtype ty)
    {
        return identity(dim4(d0), ty);
    }

    array identity(const dim_type d0,
                const dim_type d1, af_dtype ty)
    {
        return identity(dim4(d0, d1), ty);
    }

    array identity(const dim_type d0,
                const dim_type d1, const dim_type d2, af_dtype ty)
    {
        return identity(dim4(d0, d1, d2), ty);
    }

    array identity(const dim_type d0,
                const dim_type d1, const dim_type d2,
                const dim_type d3, af_dtype ty)
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
