/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/dim4.hpp>
#include <af/lapack.h>

#include <Array.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <svd.hpp>
#include <af/defines.h>

using af::dim4;
using af::dtype_traits;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::createEmptyArray;
using std::min;

template<typename T>
static inline void svd(af_array *s, af_array *u, af_array *vt,
                       const af_array in) {
    const ArrayInfo &info = getInfo(in);  // ArrayInfo is the base class which
    dim4 dims             = info.dims();
    int M                 = dims[0];
    int N                 = dims[1];

    using Tr = typename dtype_traits<T>::base_type;

    // Allocate output arrays
    Array<Tr> sA = createEmptyArray<Tr>(dim4(min(M, N)));
    Array<T> uA  = createEmptyArray<T>(dim4(M, M));
    Array<T> vtA = createEmptyArray<T>(dim4(N, N));

    svd<T, Tr>(sA, uA, vtA, getArray<T>(in));

    *s  = getHandle(sA);
    *u  = getHandle(uA);
    *vt = getHandle(vtA);
}

template<typename T>
static inline void svdInPlace(af_array *s, af_array *u, af_array *vt,
                              af_array in) {
    const ArrayInfo &info = getInfo(in);  // ArrayInfo is the base class which
    dim4 dims             = info.dims();
    int M                 = dims[0];
    int N                 = dims[1];

    using Tr = typename dtype_traits<T>::base_type;

    // Allocate output arrays
    Array<Tr> sA = createEmptyArray<Tr>(dim4(min(M, N)));
    Array<T> uA  = createEmptyArray<T>(dim4(M, M));
    Array<T> vtA = createEmptyArray<T>(dim4(N, N));

    svdInPlace<T, Tr>(sA, uA, vtA, getArray<T>(in));

    *s  = getHandle(sA);
    *u  = getHandle(uA);
    *vt = getHandle(vtA);
}

af_err af_svd(af_array *u, af_array *s, af_array *vt, const af_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        dim4 dims             = info.dims();

        ARG_ASSERT(3, (dims.ndims() >= 0 && dims.ndims() <= 2));
        af_dtype type = info.getType();

        if (dims.ndims() == 0) {
            AF_CHECK(af_create_handle(u, 0, nullptr, type));
            AF_CHECK(af_create_handle(s, 0, nullptr, type));
            AF_CHECK(af_create_handle(vt, 0, nullptr, type));
            return AF_SUCCESS;
        }

        switch (type) {
            case f64: svd<double>(s, u, vt, in); break;
            case f32: svd<float>(s, u, vt, in); break;
            case c64: svd<cdouble>(s, u, vt, in); break;
            case c32: svd<cfloat>(s, u, vt, in); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_svd_inplace(af_array *u, af_array *s, af_array *vt, af_array in) {
    try {
        const ArrayInfo &info = getInfo(in);
        dim4 dims             = info.dims();

        ARG_ASSERT(3, (dims.ndims() >= 0 && dims.ndims() <= 2));
        af_dtype type = info.getType();

        if (dims.ndims() == 0) {
            AF_CHECK(af_create_handle(u, 0, nullptr, type));
            AF_CHECK(af_create_handle(s, 0, nullptr, type));
            AF_CHECK(af_create_handle(vt, 0, nullptr, type));
            return AF_SUCCESS;
        }

        DIM_ASSERT(3, dims[0] >= dims[1]);

        switch (type) {
            case f64: svdInPlace<double>(s, u, vt, in); break;
            case f32: svdInPlace<float>(s, u, vt, in); break;
            case c64: svdInPlace<cdouble>(s, u, vt, in); break;
            case c32: svdInPlace<cfloat>(s, u, vt, in); break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
