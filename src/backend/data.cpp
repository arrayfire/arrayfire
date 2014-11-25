/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <complex>
#include <af/dim4.hpp>
#include <af/array.h>
#include <af/data.h>
#include <af/device.h>
#include <af/util.h>
#include <copy.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <handle.hpp>
#include <random.hpp>
#include <math.hpp>
#include <complex.hpp>
#include <iota.hpp>

using af::dim4;
using namespace detail;
using namespace std;

af_err af_get_data_ptr(void *data, const af_array arr)
{
    af_err ret = AF_SUCCESS;

    try {
        af_dtype type;
        af_get_type(&type, arr);
        switch(type) {
        case f32:   copyData(static_cast<float    *>(data), arr);  break;
        case c32:   copyData(static_cast<cfloat   *>(data), arr);  break;
        case f64:   copyData(static_cast<double   *>(data), arr);  break;
        case c64:   copyData(static_cast<cdouble  *>(data), arr);  break;
        case b8:    copyData(static_cast<char     *>(data), arr);  break;
        case s32:   copyData(static_cast<int      *>(data), arr);  break;
        case u32:   copyData(static_cast<unsigned *>(data), arr);  break;
        case u8:    copyData(static_cast<uchar    *>(data), arr);  break;
        default:    ret  = AF_ERR_RUNTIME;                         break;
        }
    }
    CATCHALL
        return ret;
}

//Strong Exception Guarantee
af_err af_create_array(af_array *result, const void * const data,
                       const unsigned ndims, const dim_type * const dims,
                       const af_dtype type)
{
    af_init();
    af_err ret = AF_ERR_ARG;
    af_array out;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }
        switch(type) {
        case f32:   out = createHandle(d, static_cast<const float   *>(data)); break;
        case c32:   out = createHandle(d, static_cast<const cfloat  *>(data)); break;
        case f64:   out = createHandle(d, static_cast<const double  *>(data)); break;
        case c64:   out = createHandle(d, static_cast<const cdouble *>(data)); break;
        case b8:    out = createHandle(d, static_cast<const char    *>(data)); break;
        case s32:   out = createHandle(d, static_cast<const int     *>(data)); break;
        case u32:   out = createHandle(d, static_cast<const uint    *>(data)); break;
        case u8:    out = createHandle(d, static_cast<const uchar   *>(data)); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;    break;
        }
        std::swap(*result, out);
        ret = AF_SUCCESS;
    }
    CATCHALL
        return ret;
}

//Strong Exception Guarantee
af_err af_constant(af_array *result, const double value,
                   const unsigned ndims, const dim_type * const dims,
                   const af_dtype type)
{
    af_init();
    af_err ret = AF_ERR_ARG;
    af_array out;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }
        switch(type) {
        case f32:   out = createHandle<float  >(d, value); break;
        case c32:   out = createHandle<cfloat >(d, value); break;
        case f64:   out = createHandle<double >(d, value); break;
        case c64:   out = createHandle<cdouble>(d, value); break;
        case b8:    out = createHandle<char   >(d, value); break;
        case s32:   out = createHandle<int    >(d, value); break;
        case u32:   out = createHandle<uint   >(d, value); break;
        case u8:    out = createHandle<uchar  >(d, value); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;    break;
        }
        std::swap(*result, out);
        ret = AF_SUCCESS;
    }
    CATCHALL
        return ret;
}

template<typename To, typename Ti>
static inline af_array cplx(const af_array lhs, const af_array rhs)
{
    return getHandle(*cplx<To, Ti>(getArray<Ti>(lhs), getArray<Ti>(rhs)));
}

af_err af_constant_c64(af_array *result, const void* value,
                       const unsigned ndims, const dim_type * const dims)
{
    af_init();
    af_err ret = AF_ERR_ARG;
    af_array out_real;
    af_array out_imag;
    af_array out;
    try {
        cdouble cval = *(cdouble*)value;
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }
        out_real = createHandle<double>(d, real(cval));
        out_imag = createHandle<double>(d, imag(cval));
        out = cplx<cdouble, double>(out_real, out_imag);
        std::swap(*result, out);
        ret = AF_SUCCESS;
    }
    CATCHALL
    return ret;
}

af_err af_constant_c32(af_array *result, const void* value,
                       const unsigned ndims, const dim_type * const dims)
{
    af_init();
    af_err ret = AF_ERR_ARG;
    af_array out_real;
    af_array out_imag;
    af_array out;
    try {
        cfloat cval = *(cfloat*)value;
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }
        out_real = createHandle<float>(d, real(cval));
        out_imag = createHandle<float>(d, imag(cval));
        out = cplx<cfloat, float>(out_real, out_imag);
        std::swap(*result, out);
        ret = AF_SUCCESS;
    }
    CATCHALL
    return ret;
}

//Strong Exception Guarantee
af_err af_create_handle(af_array *result, const unsigned ndims, const dim_type * const dims,
                        const af_dtype type)
{
    af_init();
    af_err ret = AF_ERR_ARG;
    af_array out;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }
        switch(type) {
        case f32:   out = createHandle<float  >(d); break;
        case c32:   out = createHandle<cfloat >(d); break;
        case f64:   out = createHandle<double >(d); break;
        case c64:   out = createHandle<cdouble>(d); break;
        case b8:    out = createHandle<char   >(d); break;
        case s32:   out = createHandle<int    >(d); break;
        case u32:   out = createHandle<uint   >(d); break;
        case u8:    out = createHandle<uchar  >(d); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;     break;
        }
        std::swap(*result, out);
        ret = AF_SUCCESS;
    }
    CATCHALL
    return ret;
}

//Strong Exception Guarantee
af_err af_copy_array(af_array *out, const af_array in)
{
    ArrayInfo info = getInfo(in);
    const unsigned ndims = info.ndims();
    const af::dim4 dims = info.dims();
    const af_dtype type = info.getType();

    af_err ret = AF_ERR_ARG;

    ret = af_create_handle(out, ndims, dims.get(), type);
    if(ret != AF_SUCCESS) {
        return ret;
    }

    try {
        switch(type) {
        case f32:   copyArray<float   >(out, in); break;
        case c32:   copyArray<cfloat  >(out, in); break;
        case f64:   copyArray<double  >(out, in); break;
        case c64:   copyArray<cdouble >(out, in); break;
        case b8:    copyArray<char    >(out, in); break;
        case s32:   copyArray<int     >(out, in); break;
        case u32:   copyArray<unsigned>(out, in); break;
        case u8:    copyArray<uchar   >(out, in); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;   break;
        }
    }
    CATCHALL
    return ret;
}

template<typename T>
static inline af_array randn_(const af::dim4 &dims)
{
    return getHandle(*randn<T>(dims));
}

template<typename T>
static inline af_array randu_(const af::dim4 &dims)
{
    return getHandle(*randu<T>(dims));
}

af_err af_randu(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type)
{
    af_init();
    af_err ret = AF_SUCCESS;
    af_array result;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
            if(d[i] < 1) {
                return AF_ERR_ARG;
            }
        }
        switch(type) {
            case f32:   result = randu_<float  >(d);    break;
            case c32:   result = randu_<cfloat >(d);    break;
            case f64:   result = randu_<double >(d);    break;
            case c64:   result = randu_<cdouble>(d);    break;
            case s32:   result = randu_<int    >(d);    break;
            case u32:   result = randu_<uint   >(d);    break;
            case u8:    result = randu_<uchar  >(d);    break;
            // Removed because of bool type. Functions implementations exist.
            //case b8:    result = randu_<char   >(d);    break;
            default:    ret    = AF_ERR_NOT_SUPPORTED; break;
        }
        if(ret == AF_SUCCESS)
            std::swap(*out, result);
    }
    CATCHALL
    return ret;
}

af_err af_randn(af_array *out, const unsigned ndims, const dim_type * const dims, const af_dtype type)
{
    af_init();
    af_err ret = AF_SUCCESS;
    af_array result;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
            if(d[i] < 1) {
                return AF_ERR_ARG;
            }
        }
        switch(type) {
            case f32:   result = randn_<float  >(d);    break;
            case c32:   result = randn_<cfloat >(d);    break;
            case f64:   result = randn_<double >(d);    break;
            case c64:   result = randn_<cdouble>(d);    break;
            default:    ret    = AF_ERR_NOT_SUPPORTED; break;
        }
        if(ret == AF_SUCCESS)
            std::swap(*out, result);
    }
    CATCHALL
    return ret;
}

af_err af_destroy_array(af_array arr)
{
    af_err ret = AF_ERR_ARG;
    try {
        af_dtype type;
        af_get_type(&type, arr);

        switch(type) {
        case f32:   destroyHandle<float   >(arr); break;
        case c32:   destroyHandle<cfloat  >(arr); break;
        case f64:   destroyHandle<double  >(arr); break;
        case c64:   destroyHandle<cdouble >(arr); break;
        case b8:    destroyHandle<char    >(arr); break;
        case s32:   destroyHandle<int     >(arr); break;
        case u32:   destroyHandle<uint    >(arr); break;
        case u8:    destroyHandle<uchar   >(arr); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;    break;
        }

        ret = AF_SUCCESS;
    }
    CATCHALL
        return ret;
}


template<typename T>
static af_array weakCopyHandle(const af_array in)
{
    detail::Array<T> *A = reinterpret_cast<detail::Array<T> *>(in);
    detail::Array<T> *out = detail::createEmptyArray<T>(af::dim4());
    *out= *A;
    return reinterpret_cast<af_array>(out);
}

af_array weakCopy(const af_array in)
{
    switch(getInfo(in).getType()) {
    case f32: return weakCopyHandle<float           >(in);
    case f64: return weakCopyHandle<double          >(in);
    case s32: return weakCopyHandle<int             >(in);
    case u32: return weakCopyHandle<unsigned int    >(in);
    case u8:  return weakCopyHandle<unsigned char   >(in);
    case c32: return weakCopyHandle<detail::cfloat  >(in);
    case c64: return weakCopyHandle<detail::cdouble >(in);
    case b8:  return weakCopyHandle<char            >(in);
    default:
        AF_ERROR("Invalid type", AF_ERR_INVALID_TYPE);
    }
}

af_err af_weak_copy(af_array *out, const af_array in)
{
    try {
        *out = weakCopy(in);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<typename T>
static inline af_array iota_(const dim4& d, const unsigned rep)
{
    return getHandle(*iota<T>(d, rep));
}

//Strong Exception Guarantee
af_err af_iota(af_array *result, const unsigned ndims, const dim_type * const dims,
               const unsigned rep, const af_dtype type)
{
    af_init();
    af_err ret = AF_ERR_ARG;
    af_array out;
    try {
        dim4 d((size_t)dims[0]);
        for(unsigned i = 1; i < ndims; i++) {
            d[i] = dims[i];
        }
        switch(type) {
        case f32:   out = iota_<float  >(d, rep); break;
        case f64:   out = iota_<double >(d, rep); break;
        case s32:   out = iota_<int    >(d, rep); break;
        case u32:   out = iota_<uint   >(d, rep); break;
        case u8:    out = iota_<uchar  >(d, rep); break;
        default:    ret = AF_ERR_NOT_SUPPORTED;  break;
        }
        std::swap(*result, out);
        ret = AF_SUCCESS;
    }
    CATCHALL
        return ret;
}

#undef INSTANTIATE
#define INSTANTIATE(fn1, fn2)                                               \
af_err fn1(bool *result, const af_array in)                                 \
{                                                                           \
    af_err ret = AF_ERR_ARG;                                                \
    try {                                                                   \
        ArrayInfo info = getInfo(in);                                       \
        *result = info.fn2();                                               \
        ret = AF_SUCCESS;                                                   \
    }                                                                       \
    CATCHALL                                                                \
    return ret;                                                             \
}

INSTANTIATE(af_is_empty       , isEmpty       )
INSTANTIATE(af_is_scalar      , isScalar      )
INSTANTIATE(af_is_row         , isRow         )
INSTANTIATE(af_is_column      , isColumn      )
INSTANTIATE(af_is_vector      , isVector      )
INSTANTIATE(af_is_complex     , isComplex     )
INSTANTIATE(af_is_real        , isReal        )
INSTANTIATE(af_is_double      , isDouble      )
INSTANTIATE(af_is_single      , isSingle      )
INSTANTIATE(af_is_realfloating, isRealFloating)
INSTANTIATE(af_is_floating    , isFloating    )
INSTANTIATE(af_is_integer     , isInteger     )
INSTANTIATE(af_is_bool        , isBool        )

#undef INSTANTIATE

af_err af_get_dims(dim_type *d0, dim_type *d1, dim_type *d2, dim_type *d3,
                   const af_array in)
{
    af_err ret = AF_ERR_ARG;
    try {
        ArrayInfo info = getInfo(in);
        *d0 = info.dims()[0];
        *d1 = info.dims()[1];
        *d2 = info.dims()[2];
        *d3 = info.dims()[3];
        ret = AF_SUCCESS;
    }
    CATCHALL
    return ret;
}

af_err af_get_numdims(unsigned *nd, const af_array in)
{
    af_err ret = AF_ERR_ARG;
    try {
        ArrayInfo info = getInfo(in);
        *nd = info.ndims();
        ret = AF_SUCCESS;
    }
    CATCHALL
    return ret;
}

template<typename T>
static inline void eval(af_array arr)
{
    getArray<T>(arr).eval();
    return;
}

af_err af_eval(af_array arr)
{
    try {
        af_dtype type = getInfo(arr).getType();
        switch (type) {
        case f32: eval<float  >(arr); break;
        case f64: eval<double >(arr); break;
        case c32: eval<cfloat >(arr); break;
        case c64: eval<cdouble>(arr); break;
        case s32: eval<int    >(arr); break;
        case u32: eval<uint   >(arr); break;
        case u8 : eval<uchar  >(arr); break;
        case b8 : eval<char   >(arr); break;
        default:
            TYPE_ERROR(0, type);
        }
    } CATCHALL;

    return AF_SUCCESS;
}
