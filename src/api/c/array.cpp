/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/half.hpp>
#include <copy.hpp>
#include <handle.hpp>
#include <platform.hpp>
#include <sparse.hpp>
#include <sparse_handle.hpp>
#include <af/sparse.h>

using af::dim4;
using arrayfire::copyData;
using arrayfire::copySparseArray;
using arrayfire::getSparseArrayBase;
using arrayfire::releaseHandle;
using arrayfire::releaseSparseHandle;
using arrayfire::retainSparseHandle;
using arrayfire::common::half;
using arrayfire::common::SparseArrayBase;
using detail::cdouble;
using detail::cfloat;
using detail::intl;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

af_err af_get_data_ptr(void *data, const af_array arr) {
    try {
        af_dtype type = getInfo(arr).getType();
        // clang-format off
        switch (type) {
            case f32: copyData(static_cast<float*   >(data), arr); break;
            case c32: copyData(static_cast<cfloat*  >(data), arr); break;
            case f64: copyData(static_cast<double*  >(data), arr); break;
            case c64: copyData(static_cast<cdouble* >(data), arr); break;
            case b8:  copyData(static_cast<char*    >(data), arr); break;
            case s32: copyData(static_cast<int*     >(data), arr); break;
            case u32: copyData(static_cast<unsigned*>(data), arr); break;
            case u8:  copyData(static_cast<uchar*   >(data), arr); break;
            case s64: copyData(static_cast<intl*    >(data), arr); break;
            case u64: copyData(static_cast<uintl*   >(data), arr); break;
            case s16: copyData(static_cast<short*   >(data), arr); break;
            case u16: copyData(static_cast<ushort*  >(data), arr); break;
            case f16: copyData(static_cast<half*    >(data), arr); break;
            default: TYPE_ERROR(1, type);
        }
        // clang-format on
    }
    CATCHALL;
    return AF_SUCCESS;
}

// Strong Exception Guarantee
af_err af_create_array(af_array *result, const void *const data,
                       const unsigned ndims, const dim_t *const dims,
                       const af_dtype type) {
    try {
        af_array out;
        AF_CHECK(af_init());

        dim4 d = verifyDims(ndims, dims);

        switch (type) {
            case f32:
                out = createHandleFromData(d, static_cast<const float *>(data));
                break;
            case c32:
                out =
                    createHandleFromData(d, static_cast<const cfloat *>(data));
                break;
            case f64:
                out =
                    createHandleFromData(d, static_cast<const double *>(data));
                break;
            case c64:
                out =
                    createHandleFromData(d, static_cast<const cdouble *>(data));
                break;
            case b8:
                out = createHandleFromData(d, static_cast<const char *>(data));
                break;
            case s32:
                out = createHandleFromData(d, static_cast<const int *>(data));
                break;
            case u32:
                out = createHandleFromData(d, static_cast<const uint *>(data));
                break;
            case u8:
                out = createHandleFromData(d, static_cast<const uchar *>(data));
                break;
            case s64:
                out = createHandleFromData(d, static_cast<const intl *>(data));
                break;
            case u64:
                out = createHandleFromData(d, static_cast<const uintl *>(data));
                break;
            case s16:
                out = createHandleFromData(d, static_cast<const short *>(data));
                break;
            case u16:
                out =
                    createHandleFromData(d, static_cast<const ushort *>(data));
                break;
            case f16:
                out = createHandleFromData(d, static_cast<const half *>(data));
                break;
            default: TYPE_ERROR(4, type);
        }
        std::swap(*result, out);
    }
    CATCHALL
    return AF_SUCCESS;
}

// Strong Exception Guarantee
af_err af_create_handle(af_array *result, const unsigned ndims,
                        const dim_t *const dims, const af_dtype type) {
    try {
        AF_CHECK(af_init());

        if (ndims > 0) { ARG_ASSERT(2, ndims > 0 && dims != NULL); }

        dim4 d(0);
        for (unsigned i = 0; i < ndims; i++) { d[i] = dims[i]; }

        af_array out = createHandle(d, type);
        std::swap(*result, out);
    }
    CATCHALL
    return AF_SUCCESS;
}

// Strong Exception Guarantee
af_err af_copy_array(af_array *out, const af_array in) {
    try {
        const ArrayInfo &info = getInfo(in, false);
        const af_dtype type   = info.getType();

        af_array res = 0;
        if (info.isSparse()) {
            const SparseArrayBase sbase = getSparseArrayBase(in);
            if (info.ndims() == 0) {
                return af_create_sparse_array_from_ptr(
                    out, info.dims()[0], info.dims()[1], 0, nullptr, nullptr,
                    nullptr, type, sbase.getStorage(), afDevice);
            }
            switch (type) {
                case f32: res = copySparseArray<float>(in); break;
                case f64: res = copySparseArray<double>(in); break;
                case c32: res = copySparseArray<cfloat>(in); break;
                case c64: res = copySparseArray<cdouble>(in); break;
                default: TYPE_ERROR(0, type);
            }

        } else {
            if (info.ndims() == 0) {
                return af_create_handle(out, 0, nullptr, type);
            }
            switch (type) {
                case f32: res = copyArray<float>(in); break;
                case c32: res = copyArray<cfloat>(in); break;
                case f64: res = copyArray<double>(in); break;
                case c64: res = copyArray<cdouble>(in); break;
                case b8: res = copyArray<char>(in); break;
                case s32: res = copyArray<int>(in); break;
                case u32: res = copyArray<uint>(in); break;
                case u8: res = copyArray<uchar>(in); break;
                case s64: res = copyArray<intl>(in); break;
                case u64: res = copyArray<uintl>(in); break;
                case s16: res = copyArray<short>(in); break;
                case u16: res = copyArray<ushort>(in); break;
                case f16: res = copyArray<half>(in); break;
                default: TYPE_ERROR(1, type);
            }
        }
        std::swap(*out, res);
    }
    CATCHALL
    return AF_SUCCESS;
}

// Strong Exception Guarantee
af_err af_get_data_ref_count(int *use_count, const af_array in) {
    try {
        const ArrayInfo &info = getInfo(in, false, false);
        const af_dtype type   = info.getType();

        int res;
        switch (type) {
            case f32: res = getArray<float>(in).useCount(); break;
            case c32: res = getArray<cfloat>(in).useCount(); break;
            case f64: res = getArray<double>(in).useCount(); break;
            case c64: res = getArray<cdouble>(in).useCount(); break;
            case b8: res = getArray<char>(in).useCount(); break;
            case s32: res = getArray<int>(in).useCount(); break;
            case u32: res = getArray<uint>(in).useCount(); break;
            case u8: res = getArray<uchar>(in).useCount(); break;
            case s64: res = getArray<intl>(in).useCount(); break;
            case u64: res = getArray<uintl>(in).useCount(); break;
            case s16: res = getArray<short>(in).useCount(); break;
            case u16: res = getArray<ushort>(in).useCount(); break;
            case f16: res = getArray<half>(in).useCount(); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*use_count, res);
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_release_array(af_array arr) {
    try {
        if (arr == 0) { return AF_SUCCESS; }
        const ArrayInfo &info = getInfo(arr, false, false);
        af_dtype type         = info.getType();

        if (info.isSparse()) {
            switch (type) {
                case f32: releaseSparseHandle<float>(arr); break;
                case f64: releaseSparseHandle<double>(arr); break;
                case c32: releaseSparseHandle<cfloat>(arr); break;
                case c64: releaseSparseHandle<cdouble>(arr); break;
                default: TYPE_ERROR(0, type);
            }
        } else {
            switch (type) {
                case f32: releaseHandle<float>(arr); break;
                case c32: releaseHandle<cfloat>(arr); break;
                case f64: releaseHandle<double>(arr); break;
                case c64: releaseHandle<cdouble>(arr); break;
                case b8: releaseHandle<char>(arr); break;
                case s32: releaseHandle<int>(arr); break;
                case u32: releaseHandle<uint>(arr); break;
                case u8: releaseHandle<uchar>(arr); break;
                case s64: releaseHandle<intl>(arr); break;
                case u64: releaseHandle<uintl>(arr); break;
                case s16: releaseHandle<short>(arr); break;
                case u16: releaseHandle<ushort>(arr); break;
                case f16: releaseHandle<half>(arr); break;
                default: TYPE_ERROR(0, type);
            }
        }
    }
    CATCHALL

    return AF_SUCCESS;
}

af_err af_retain_array(af_array *out, const af_array in) {
    try {
        *out = retain(in);
    }
    CATCHALL;
    return AF_SUCCESS;
}

template<typename T>
void write_array(af_array arr, const T *const data, const size_t bytes,
                 af_source src) {
    if (src == afHost) {
        writeHostDataArray(getArray<T>(arr), data, bytes);
    } else {
        writeDeviceDataArray(getArray<T>(arr), data, bytes);
    }
}

af_err af_write_array(af_array arr, const void *data, const size_t bytes,
                      af_source src) {
    if (bytes == 0) { return AF_SUCCESS; }
    try {
        af_dtype type = getInfo(arr).getType();
        ARG_ASSERT(1, (data != nullptr));
        ARG_ASSERT(3, (src == afHost || src == afDevice));
        // FIXME ArrayInfo class no bytes method, hence commented
        // DIM_ASSERT(2, bytes <= getInfo(arr).bytes());

        switch (type) {
            case f32:
                write_array(arr, static_cast<const float *>(data), bytes, src);
                break;
            case c32:
                write_array(arr, static_cast<const cfloat *>(data), bytes, src);
                break;
            case f64:
                write_array(arr, static_cast<const double *>(data), bytes, src);
                break;
            case c64:
                write_array(arr, static_cast<const cdouble *>(data), bytes,
                            src);
                break;
            case b8:
                write_array(arr, static_cast<const char *>(data), bytes, src);
                break;
            case s32:
                write_array(arr, static_cast<const int *>(data), bytes, src);
                break;
            case u32:
                write_array(arr, static_cast<const uint *>(data), bytes, src);
                break;
            case u8:
                write_array(arr, static_cast<const uchar *>(data), bytes, src);
                break;
            case s64:
                write_array(arr, static_cast<const intl *>(data), bytes, src);
                break;
            case u64:
                write_array(arr, static_cast<const uintl *>(data), bytes, src);
                break;
            case s16:
                write_array(arr, static_cast<const short *>(data), bytes, src);
                break;
            case u16:
                write_array(arr, static_cast<const ushort *>(data), bytes, src);
                break;
            case f16:
                write_array(arr, static_cast<const half *>(data), bytes, src);
                break;
            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_get_elements(dim_t *elems, const af_array arr) {
    try {
        // Do not check for device mismatch
        *elems = getInfo(arr, false, false).elements();
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_get_type(af_dtype *type, const af_array arr) {
    try {
        // Do not check for device mismatch
        *type = getInfo(arr, false, false).getType();
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_get_dims(dim_t *d0, dim_t *d1, dim_t *d2, dim_t *d3,
                   const af_array in) {
    try {
        // Do not check for device mismatch
        const ArrayInfo &info = getInfo(in, false, false);
        *d0                   = info.dims()[0];
        *d1                   = info.dims()[1];
        *d2                   = info.dims()[2];
        *d3                   = info.dims()[3];
    }
    CATCHALL
    return AF_SUCCESS;
}

af_err af_get_numdims(unsigned *nd, const af_array in) {
    try {
        // Do not check for device mismatch
        const ArrayInfo &info = getInfo(in, false, false);
        *nd                   = info.ndims();
    }
    CATCHALL
    return AF_SUCCESS;
}

#undef INSTANTIATE
#define INSTANTIATE(fn1, fn2)                                  \
    af_err fn1(bool *result, const af_array in) {              \
        try {                                                  \
            const ArrayInfo &info = getInfo(in, false, false); \
            *result               = info.fn2();                \
        }                                                      \
        CATCHALL                                               \
        return AF_SUCCESS;                                     \
    }

INSTANTIATE(af_is_empty, isEmpty)
INSTANTIATE(af_is_scalar, isScalar)
INSTANTIATE(af_is_row, isRow)
INSTANTIATE(af_is_column, isColumn)
INSTANTIATE(af_is_vector, isVector)
INSTANTIATE(af_is_complex, isComplex)
INSTANTIATE(af_is_real, isReal)
INSTANTIATE(af_is_double, isDouble)
INSTANTIATE(af_is_single, isSingle)
INSTANTIATE(af_is_half, isHalf)
INSTANTIATE(af_is_realfloating, isRealFloating)
INSTANTIATE(af_is_floating, isFloating)
INSTANTIATE(af_is_integer, isInteger)
INSTANTIATE(af_is_bool, isBool)
INSTANTIATE(af_is_sparse, isSparse)

#undef INSTANTIATE

template<typename T>
inline void getScalar(T *out, const af_array &arr) {
    out[0] = getScalar<T>(getArray<T>(arr));
}

af_err af_get_scalar(void *output_value, const af_array arr) {
    try {
        ARG_ASSERT(0, (output_value != NULL));

        const ArrayInfo &info = getInfo(arr);
        const af_dtype type   = info.getType();

        switch (type) {
            case f32:
                getScalar<float>(reinterpret_cast<float *>(output_value), arr);
                break;
            case f64:
                getScalar<double>(reinterpret_cast<double *>(output_value),
                                  arr);
                break;
            case b8:
                getScalar<char>(reinterpret_cast<char *>(output_value), arr);
                break;
            case s32:
                getScalar<int>(reinterpret_cast<int *>(output_value), arr);
                break;
            case u32:
                getScalar<uint>(reinterpret_cast<uint *>(output_value), arr);
                break;
            case u8:
                getScalar<uchar>(reinterpret_cast<uchar *>(output_value), arr);
                break;
            case s64:
                getScalar<intl>(reinterpret_cast<intl *>(output_value), arr);
                break;
            case u64:
                getScalar<uintl>(reinterpret_cast<uintl *>(output_value), arr);
                break;
            case s16:
                getScalar<short>(reinterpret_cast<short *>(output_value), arr);
                break;
            case u16:
                getScalar<ushort>(reinterpret_cast<ushort *>(output_value),
                                  arr);
                break;
            case c32:
                getScalar<cfloat>(reinterpret_cast<cfloat *>(output_value),
                                  arr);
                break;
            case c64:
                getScalar<cdouble>(reinterpret_cast<cdouble *>(output_value),
                                   arr);
                break;
            case f16:
                getScalar<half>(static_cast<half *>(output_value), arr);
                break;
            default: TYPE_ERROR(4, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
