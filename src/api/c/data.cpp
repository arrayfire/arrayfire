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
#include <range.hpp>
#include <iota.hpp>
#include <identity.hpp>
#include <diagonal.hpp>
#include <triangle.hpp>

using af::dim4;
using namespace detail;
using namespace std;

static inline dim4 verifyDims(const unsigned ndims, const dim_t * const dims)
{

    DIM_ASSERT(1, ndims >= 1);

    dim4 d(1, 1, 1, 1);

    for(unsigned i = 0; i < ndims; i++) {
        d[i] = dims[i];
        DIM_ASSERT(2, dims[i] >= 1);
    }

    return d;
}

af_err af_get_data_ptr(void *data, const af_array arr)
{
    try {
        af_dtype type = getInfo(arr).getType();
        switch(type) {
        case f32:   copyData(static_cast<float    *>(data), arr);  break;
        case c32:   copyData(static_cast<cfloat   *>(data), arr);  break;
        case f64:   copyData(static_cast<double   *>(data), arr);  break;
        case c64:   copyData(static_cast<cdouble  *>(data), arr);  break;
        case b8:    copyData(static_cast<char     *>(data), arr);  break;
        case s32:   copyData(static_cast<int      *>(data), arr);  break;
        case u32:   copyData(static_cast<unsigned *>(data), arr);  break;
        case u8:    copyData(static_cast<uchar    *>(data), arr);  break;
        case s64:   copyData(static_cast<intl     *>(data), arr);  break;
        case u64:   copyData(static_cast<uintl    *>(data), arr);  break;
        default:    TYPE_ERROR(1, type);
        }
    }
    CATCHALL
        return AF_SUCCESS;
}

//Strong Exception Guarantee
af_err af_create_array(af_array *result, const void * const data,
                       const unsigned ndims, const dim_t * const dims,
                       const af_dtype type)
{
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d = verifyDims(ndims, dims);

        switch(type) {
        case f32:   out = createHandleFromData(d, static_cast<const float   *>(data)); break;
        case c32:   out = createHandleFromData(d, static_cast<const cfloat  *>(data)); break;
        case f64:   out = createHandleFromData(d, static_cast<const double  *>(data)); break;
        case c64:   out = createHandleFromData(d, static_cast<const cdouble *>(data)); break;
        case b8:    out = createHandleFromData(d, static_cast<const char    *>(data)); break;
        case s32:   out = createHandleFromData(d, static_cast<const int     *>(data)); break;
        case u32:   out = createHandleFromData(d, static_cast<const uint    *>(data)); break;
        case u8:    out = createHandleFromData(d, static_cast<const uchar   *>(data)); break;
        case s64:   out = createHandleFromData(d, static_cast<const intl    *>(data)); break;
        case u64:   out = createHandleFromData(d, static_cast<const uintl   *>(data)); break;
        default:    TYPE_ERROR(4, type);
        }
        std::swap(*result, out);
    }
    CATCHALL
        return AF_SUCCESS;
}

//Strong Exception Guarantee
af_err af_constant(af_array *result, const double value,
                   const unsigned ndims, const dim_t * const dims,
                   const af_dtype type)
{
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d = verifyDims(ndims, dims);

        switch(type) {
        case f32:   out = createHandleFromValue<float  >(d, value); break;
        case c32:   out = createHandleFromValue<cfloat >(d, value); break;
        case f64:   out = createHandleFromValue<double >(d, value); break;
        case c64:   out = createHandleFromValue<cdouble>(d, value); break;
        case b8:    out = createHandleFromValue<char   >(d, value); break;
        case s32:   out = createHandleFromValue<int    >(d, value); break;
        case u32:   out = createHandleFromValue<uint   >(d, value); break;
        case u8:    out = createHandleFromValue<uchar  >(d, value); break;
        case s64:   out = createHandleFromValue<intl   >(d, value); break;
        case u64:   out = createHandleFromValue<uintl  >(d, value); break;
        default:    TYPE_ERROR(4, type);
        }
        std::swap(*result, out);
    }
    CATCHALL
        return AF_SUCCESS;
}

template<typename To, typename Ti>
static inline af_array createCplx(dim4 dims, const Ti real, const Ti imag)
{
    To cval = scalar<To, Ti>(real, imag);
    af_array out = getHandle(createValueArray<To>(dims, cval));
    return out;
}

af_err af_constant_complex(af_array *result, const double real, const double imag,
                           const unsigned ndims, const dim_t * const dims, af_dtype type)
{
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d = verifyDims(ndims, dims);

        switch (type) {
        case c32: out = createCplx<cfloat , float >(d, real, imag); break;
        case c64: out = createCplx<cdouble, double>(d, real, imag); break;
        default:   TYPE_ERROR(5, type);
        }

        std::swap(*result, out);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_constant_long(af_array *result, const intl val,
                        const unsigned ndims, const dim_t * const dims)
{
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d = verifyDims(ndims, dims);

        out = getHandle(createValueArray<intl>(d, val));

        std::swap(*result, out);
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_constant_ulong(af_array *result, const uintl val,
                         const unsigned ndims, const dim_t * const dims)
{
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d = verifyDims(ndims, dims);
        out = getHandle(createValueArray<uintl>(d, val));

        std::swap(*result, out);
    } CATCHALL;

    return AF_SUCCESS;
}

//Strong Exception Guarantee
af_err af_create_handle(af_array *result, const unsigned ndims, const dim_t * const dims,
                        const af_dtype type)
{
    try {
        af_array out;
        AF_CHECK(af_init());

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
        case s64:   out = createHandle<intl   >(d); break;
        case u64:   out = createHandle<uintl  >(d); break;
        default:    TYPE_ERROR(3, type);
        }
        std::swap(*result, out);
    }
    CATCHALL
    return AF_SUCCESS;
}

//Strong Exception Guarantee
af_err af_copy_array(af_array *out, const af_array in)
{
    try {
        ArrayInfo info = getInfo(in);
        const af_dtype type = info.getType();

        af_array res;
        switch(type) {
        case f32:   res = copyArray<float   >(in); break;
        case c32:   res = copyArray<cfloat  >(in); break;
        case f64:   res = copyArray<double  >(in); break;
        case c64:   res = copyArray<cdouble >(in); break;
        case b8:    res = copyArray<char    >(in); break;
        case s32:   res = copyArray<int     >(in); break;
        case u32:   res = copyArray<uint    >(in); break;
        case u8:    res = copyArray<uchar   >(in); break;
        case s64:   res = copyArray<intl    >(in); break;
        case u64:   res = copyArray<uintl   >(in); break;
        default:    TYPE_ERROR(1, type);
        }
        std::swap(*out, res);
    }
    CATCHALL
    return AF_SUCCESS;
}

template<typename T>
static inline af_array randn_(const af::dim4 &dims)
{
    return getHandle(randn<T>(dims));
}

template<typename T>
static inline af_array randu_(const af::dim4 &dims)
{
    return getHandle(randu<T>(dims));
}

template<typename T>
static inline af_array identity_(const af::dim4 &dims)
{
    return getHandle(detail::identity<T>(dims));
}

af_err af_randu(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    try {
        af_array result;
        AF_CHECK(af_init());

        dim4 d = verifyDims(ndims, dims);

        switch(type) {
        case f32:   result = randu_<float  >(d);    break;
        case c32:   result = randu_<cfloat >(d);    break;
        case f64:   result = randu_<double >(d);    break;
        case c64:   result = randu_<cdouble>(d);    break;
        case s32:   result = randu_<int    >(d);    break;
        case u32:   result = randu_<uint   >(d);    break;
        case u8:    result = randu_<uchar  >(d);    break;
        case b8:    result = randu_<char  >(d);    break;
        default:    TYPE_ERROR(3, type);
        }
        std::swap(*out, result);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_randn(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    try {
        af_array result;
        AF_CHECK(af_init());

        dim4 d = verifyDims(ndims, dims);

        switch(type) {
        case f32:   result = randn_<float  >(d);    break;
        case c32:   result = randn_<cfloat >(d);    break;
        case f64:   result = randn_<double >(d);    break;
        case c64:   result = randn_<cdouble>(d);    break;
        default:    TYPE_ERROR(3, type);
        }
        std::swap(*out, result);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_set_seed(const uintl seed)
{
    try {
        setSeed(seed);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_seed(uintl *seed)
{
    try {
        *seed = getSeed();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_identity(af_array *out, const unsigned ndims, const dim_t * const dims, const af_dtype type)
{
    try {
        af_array result;
        AF_CHECK(af_init());

        dim4 d = verifyDims(ndims, dims);

        switch(type) {
        case f32:   result = identity_<float  >(d);    break;
        case c32:   result = identity_<cfloat >(d);    break;
        case f64:   result = identity_<double >(d);    break;
        case c64:   result = identity_<cdouble>(d);    break;
        case s32:   result = identity_<int    >(d);    break;
        case u32:   result = identity_<uint   >(d);    break;
        case u8:    result = identity_<uchar  >(d);    break;
        case u64:   result = identity_<uintl  >(d);    break;
        case s64:   result = identity_<intl   >(d);    break;
            // Removed because of bool type. Functions implementations exist.
        case b8:    result = identity_<char   >(d);    break;
        default:    TYPE_ERROR(3, type);
        }
        std::swap(*out, result);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_release_array(af_array arr)
{
    try {
        af_dtype type = getInfo(arr).getType();

        switch(type) {
        case f32:   releaseHandle<float   >(arr); break;
        case c32:   releaseHandle<cfloat  >(arr); break;
        case f64:   releaseHandle<double  >(arr); break;
        case c64:   releaseHandle<cdouble >(arr); break;
        case b8:    releaseHandle<char    >(arr); break;
        case s32:   releaseHandle<int     >(arr); break;
        case u32:   releaseHandle<uint    >(arr); break;
        case u8:    releaseHandle<uchar   >(arr); break;
        case s64:   releaseHandle<intl    >(arr); break;
        case u64:   releaseHandle<uintl   >(arr); break;
        default:    TYPE_ERROR(0, type);
        }
    }
    CATCHALL

    return AF_SUCCESS;
}


template<typename T>
static af_array retainHandle(const af_array in)
{
    detail::Array<T> *A = reinterpret_cast<detail::Array<T> *>(in);
    detail::Array<T> *out = detail::initArray<T>();
    *out= *A;
    return reinterpret_cast<af_array>(out);
}

af_array retain(const af_array in)
{
    af_dtype ty = getInfo(in).getType();
    switch(ty) {
    case f32: return retainHandle<float           >(in);
    case f64: return retainHandle<double          >(in);
    case s32: return retainHandle<int             >(in);
    case u32: return retainHandle<uint            >(in);
    case u8:  return retainHandle<uchar           >(in);
    case c32: return retainHandle<detail::cfloat  >(in);
    case c64: return retainHandle<detail::cdouble >(in);
    case b8:  return retainHandle<char            >(in);
    case s64: return retainHandle<intl            >(in);
    case u64: return retainHandle<uintl           >(in);
    default:
        TYPE_ERROR(1, ty);
    }
}

af_err af_retain_array(af_array *out, const af_array in)
{
    try {
        *out = retain(in);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<typename T>
static inline af_array range_(const dim4& d, const int seq_dim)
{
    return getHandle(range<T>(d, seq_dim));
}

//Strong Exception Guarantee
af_err af_range(af_array *result, const unsigned ndims, const dim_t * const dims,
               const int seq_dim, const af_dtype type)
{
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d = verifyDims(ndims, dims);

        switch(type) {
        case f32:   out = range_<float  >(d, seq_dim); break;
        case f64:   out = range_<double >(d, seq_dim); break;
        case s32:   out = range_<int    >(d, seq_dim); break;
        case u32:   out = range_<uint   >(d, seq_dim); break;
        case u8:    out = range_<uchar  >(d, seq_dim); break;
        default:    TYPE_ERROR(4, type);
        }
        std::swap(*result, out);
    }
    CATCHALL
    return AF_SUCCESS;
}

template<typename T>
static inline af_array iota_(const dim4 &dims, const dim4 &tile_dims)
{
    return getHandle(iota<T>(dims, tile_dims));
}

//Strong Exception Guarantee
af_err af_iota(af_array *result, const unsigned ndims, const dim_t * const dims,
               const unsigned t_ndims, const dim_t * const tdims, const af_dtype type)
{
    try {
        af_array out;
        AF_CHECK(af_init());

        DIM_ASSERT(1, ndims > 0 && ndims <= 4);
        DIM_ASSERT(3, t_ndims > 0 && t_ndims <= 4);
        dim4 d;
        dim4 t;
        for(unsigned i = 0; i < 4; i++) {
            d[i] = dims[i];
            DIM_ASSERT(2, d[i] >= 1);
        }
        for(unsigned i = 0; i < 4; i++) {
            t[i] = tdims[i];
            DIM_ASSERT(4, t[i] >= 1);
        }

        switch(type) {
        case f32:   out = iota_<float  >(d, t); break;
        case f64:   out = iota_<double >(d, t); break;
        case s32:   out = iota_<int    >(d, t); break;
        case u32:   out = iota_<uint   >(d, t); break;
        case u8:    out = iota_<uchar  >(d, t); break;
        default:    TYPE_ERROR(4, type);
        }
        std::swap(*result, out);
    }
    CATCHALL
    return AF_SUCCESS;
}

#undef INSTANTIATE
#define INSTANTIATE(fn1, fn2)                   \
    af_err fn1(bool *result, const af_array in) \
    {                                           \
        try {                                   \
            ArrayInfo info = getInfo(in);       \
            *result = info.fn2();               \
        }                                       \
        CATCHALL                                \
            return AF_SUCCESS;                  \
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

af_err af_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3,
                   const af_array in)
{
    try {
        ArrayInfo info = getInfo(in);
        *d0 = info.dims()[0];
        *d1 = info.dims()[1];
        *d2 = info.dims()[2];
        *d3 = info.dims()[3];
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_get_numdims(unsigned *nd, const af_array in)
{
    try {
        ArrayInfo info = getInfo(in);
        *nd = info.ndims();
    }
    CATCHALL
    return AF_SUCCESS;
}

template<typename T>
static inline void eval(af_array arr)
{
    evalArray(getArray<T>(arr));
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
        case s64: eval<intl   >(arr); break;
        case u64: eval<uintl  >(arr); break;
        default:
            TYPE_ERROR(0, type);
        }
    } CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
static inline af_array diagCreate(const af_array in, const int num)
{
    return getHandle(diagCreate<T>(getArray<T>(in), num));
}

template<typename T>
static inline af_array diagExtract(const af_array in, const int num)
{
    return getHandle(diagExtract<T>(getArray<T>(in), num));
}

af_err af_diag_create(af_array *out, const af_array in, const int num)
{
    try {
        ArrayInfo in_info = getInfo(in);
        DIM_ASSERT(1, in_info.ndims() <= 2);
        af_dtype type = in_info.getType();

        af_array result;
        switch(type) {
        case f32:   result = diagCreate<float  >(in, num);    break;
        case c32:   result = diagCreate<cfloat >(in, num);    break;
        case f64:   result = diagCreate<double >(in, num);    break;
        case c64:   result = diagCreate<cdouble>(in, num);    break;
        case s32:   result = diagCreate<int    >(in, num);    break;
        case u32:   result = diagCreate<uint   >(in, num);    break;
        case s64:   result = diagCreate<intl   >(in, num);    break;
        case u64:   result = diagCreate<uintl  >(in, num);    break;
        case u8:    result = diagCreate<uchar  >(in, num);    break;
            // Removed because of bool type. Functions implementations exist.
        case b8:    result = diagCreate<char   >(in, num);    break;
        default:    TYPE_ERROR(1, type);
        }

        std::swap(*out, result);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_diag_extract(af_array *out, const af_array in, const int num)
{

    try {
        ArrayInfo in_info = getInfo(in);
        DIM_ASSERT(1, in_info.ndims() >= 2);
        af_dtype type = in_info.getType();

        af_array result;
        switch(type) {
        case f32:   result = diagExtract<float  >(in, num);    break;
        case c32:   result = diagExtract<cfloat >(in, num);    break;
        case f64:   result = diagExtract<double >(in, num);    break;
        case c64:   result = diagExtract<cdouble>(in, num);    break;
        case s32:   result = diagExtract<int    >(in, num);    break;
        case u32:   result = diagExtract<uint   >(in, num);    break;
        case s64:   result = diagExtract<intl   >(in, num);    break;
        case u64:   result = diagExtract<uintl  >(in, num);    break;
        case u8:    result = diagExtract<uchar  >(in, num);    break;
            // Removed because of bool type. Functions implementations exist.
        case b8:    result = diagExtract<char   >(in, num);    break;
        default:    TYPE_ERROR(1, type);
        }

        std::swap(*out, result);
    } CATCHALL;

    return AF_SUCCESS;
}

template<typename T>
void write_array(af_array arr, const T * const data, const size_t bytes, af_source src)
{
    if(src == afHost) {
        writeHostDataArray(getWritableArray<T>(arr), data, bytes);
    } else {
        writeDeviceDataArray(getWritableArray<T>(arr), data, bytes);
    }
    return;
}

af_err af_write_array(af_array arr, const void *data, const size_t bytes, af_source src)
{
    try {
        af_dtype type = getInfo(arr).getType();
        //DIM_ASSERT(2, bytes <= getInfo(arr).bytes());

        switch(type) {
        case f32:   write_array(arr, static_cast<const float   *>(data), bytes, src); break;
        case c32:   write_array(arr, static_cast<const cfloat  *>(data), bytes, src); break;
        case f64:   write_array(arr, static_cast<const double  *>(data), bytes, src); break;
        case c64:   write_array(arr, static_cast<const cdouble *>(data), bytes, src); break;
        case b8:    write_array(arr, static_cast<const char    *>(data), bytes, src); break;
        case s32:   write_array(arr, static_cast<const int     *>(data), bytes, src); break;
        case u32:   write_array(arr, static_cast<const uint    *>(data), bytes, src); break;
        case u8:    write_array(arr, static_cast<const uchar   *>(data), bytes, src); break;
        case s64:   write_array(arr, static_cast<const intl    *>(data), bytes, src); break;
        case u64:   write_array(arr, static_cast<const uintl   *>(data), bytes, src); break;
        default:    TYPE_ERROR(4, type);
        }
    }
    CATCHALL
        return AF_SUCCESS;
}

template<typename T, bool is_upper>
af_array triangle(const af_array in, bool is_unit_diag)
{
    if (is_unit_diag)
        return getHandle(triangle<T, is_upper,  true>(getArray<T>(in)));
    else
        return getHandle(triangle<T, is_upper, false>(getArray<T>(in)));
}

af_err af_lower(af_array *out, const af_array in, bool is_unit_diag)
{
    try {
        af_dtype type = getInfo(in).getType();
        af_array res;
        switch(type) {
        case f32: res = triangle<float   , false>(in, is_unit_diag); break;
        case f64: res = triangle<double  , false>(in, is_unit_diag); break;
        case c32: res = triangle<cfloat  , false>(in, is_unit_diag); break;
        case c64: res = triangle<cdouble , false>(in, is_unit_diag); break;
        case s32: res = triangle<int     , false>(in, is_unit_diag); break;
        case s64: res = triangle<intl    , false>(in, is_unit_diag); break;
        case u32: res = triangle<uint    , false>(in, is_unit_diag); break;
        case u64: res = triangle<uintl   , false>(in, is_unit_diag); break;
        case u8 : res = triangle<uchar   , false>(in, is_unit_diag); break;
        case b8 : res = triangle<char    , false>(in, is_unit_diag); break;
        }
        std::swap(*out, res);
    }
    CATCHALL
        return AF_SUCCESS;
}


af_err af_upper(af_array *out, const af_array in, bool is_unit_diag)
{
    try {
        af_dtype type = getInfo(in).getType();
        af_array res;
        switch(type) {
        case f32: res = triangle<float   , true>(in, is_unit_diag); break;
        case f64: res = triangle<double  , true>(in, is_unit_diag); break;
        case c32: res = triangle<cfloat  , true>(in, is_unit_diag); break;
        case c64: res = triangle<cdouble , true>(in, is_unit_diag); break;
        case s32: res = triangle<int     , true>(in, is_unit_diag); break;
        case s64: res = triangle<intl    , true>(in, is_unit_diag); break;
        case u32: res = triangle<uint    , true>(in, is_unit_diag); break;
        case u64: res = triangle<uintl   , true>(in, is_unit_diag); break;
        case u8 : res = triangle<uchar   , true>(in, is_unit_diag); break;
        case b8 : res = triangle<char    , true>(in, is_unit_diag); break;
        }
        std::swap(*out, res);
    }
    CATCHALL
        return AF_SUCCESS;
}
