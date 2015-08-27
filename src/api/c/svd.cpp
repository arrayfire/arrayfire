/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/array.h>
#include <af/lapack.h>

#include <af/util.h>
#include <af/defines.h>
#include <err_common.hpp>
#include <backend.hpp>
#include <Array.hpp>
#include <handle.hpp>
#include <svd.hpp>

using namespace detail;

template <typename T>
static inline void svd(af_array *s, af_array *u, af_array *vt, const af_array in)
{
    ArrayInfo info = getInfo(in);  // ArrayInfo is the base class which
    af::dim4 dims = info.dims();
    int M = dims[0];
    int N = dims[1];

    typedef typename af::dtype_traits<T>::base_type Tr;

    //Allocate output arrays
    Array<Tr> sA  = createEmptyArray<Tr>(af::dim4(min(M, N)));
    Array<T > uA  = createEmptyArray<T >(af::dim4(M, M));
    Array<T > vtA = createEmptyArray<T >(af::dim4(N, N));

    svd<T, Tr>(sA, uA, vtA, getArray<T>(in));

    *s = getHandle(sA);
    *u = getHandle(uA);
    *vt = getHandle(vtA);
}

template <typename T>
static inline void svdInPlace(af_array *s, af_array *u, af_array *vt, af_array in)
{
    ArrayInfo info = getInfo(in);  // ArrayInfo is the base class which
    af::dim4 dims = info.dims();
    int M = dims[0];
    int N = dims[1];

    typedef typename af::dtype_traits<T>::base_type Tr;

    //Allocate output arrays
    Array<Tr> sA  = createEmptyArray<Tr>(af::dim4(min(M, N)));
    Array<T > uA  = createEmptyArray<T >(af::dim4(M, M));
    Array<T > vtA = createEmptyArray<T >(af::dim4(N, N));

    svdInPlace<T, Tr>(sA, uA, vtA, getWritableArray<T>(in));

    *s = getHandle(sA);
    *u = getHandle(uA);
    *vt = getHandle(vtA);
}

af_err af_svd(af_array *u, af_array *s, af_array *vt, const af_array in)
{
    try {
        ArrayInfo info = getInfo(in);
        af::dim4 dims = info.dims();

        ARG_ASSERT(3, (dims.ndims() >= 0 && dims.ndims() <= 3));
        af_dtype type = info.getType();

        switch (type) {
        case f64:
            svd<double>(s, u, vt, in);
            break;
        case f32:
            svd<float>(s, u, vt, in);
            break;
        case c64:
            svd<cdouble>(s, u, vt, in);
            break;
        case c32:
            svd<cfloat>(s, u, vt, in);
            break;
        default:
            TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err af_svd_inplace(af_array *u, af_array *s, af_array *vt, af_array in)
{
    try {
        ArrayInfo info = getInfo(in);
        af::dim4 dims = info.dims();

        DIM_ASSERT(3, dims[0] <= dims[1]);
        ARG_ASSERT(3, (dims.ndims() >= 0 && dims.ndims() <= 3));

        af_dtype type = info.getType();

        switch (type) {
        case f64:
            svdInPlace<double>(s, u, vt, in);
            break;
        case f32:
            svdInPlace<float>(s, u, vt, in);
            break;
        case c64:
            svdInPlace<cdouble>(s, u, vt, in);
            break;
        case c32:
            svdInPlace<cfloat>(s, u, vt, in);
            break;
        default:
            TYPE_ERROR(1, type);
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}
