/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/defines.h>
#include <af/arith.h>
#include <sparse_handle.hpp>
#include <sparse.hpp>
#include <ArrayInfo.hpp>
#include <optypes.hpp>

#include <cast.hpp>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>

using namespace detail;

static af_array cast(const af_array in, const af_dtype type)
{
    const ArrayInfo info = getInfo(in);

    if (info.getType() == type) {
        return retain(in);
    }

    switch (type) {
    case f32: return getHandle(castArray<float   >(in));
    case f64: return getHandle(castArray<double  >(in));
    case c32: return getHandle(castArray<cfloat  >(in));
    case c64: return getHandle(castArray<cdouble >(in));
    case s32: return getHandle(castArray<int     >(in));
    case u32: return getHandle(castArray<uint    >(in));
    case u8 : return getHandle(castArray<uchar   >(in));
    case b8 : return getHandle(castArray<char    >(in));
    case s64: return getHandle(castArray<intl    >(in));
    case u64: return getHandle(castArray<uintl   >(in));
    case s16: return getHandle(castArray<short   >(in));
    case u16: return getHandle(castArray<ushort  >(in));
    default: TYPE_ERROR(2, type);
    }
}

template<typename T>
static af_array castSparseValues(const af_array in, const af_dtype type)
{
    using namespace common;
    const SparseArray<T> sparse = getSparseArray<T>(in);
    Array<T> values = castArray<T>(getHandle(sparse.getValues()));
    return getHandle(createArrayDataSparseArray(sparse.dims(), values,
                                                sparse.getRowIdx(), sparse.getColIdx(),
                                                sparse.getStorage()
                                               )
                    );
}

static af_array castSparse(const af_array in, const af_dtype type)
{
    using namespace common;

    const SparseArrayBase info = getSparseArrayBase(in);

    if (info.getType() == type) {
        return retain(in);
    }

    switch (type) {
    case f32: return castSparseValues<float  >(in, type);
    case f64: return castSparseValues<double >(in, type);
    case c32: return castSparseValues<cfloat >(in, type);
    case c64: return castSparseValues<cdouble>(in, type);
    default: TYPE_ERROR(2, type);
    }
}

af_err af_cast(af_array *out, const af_array in, const af_dtype type)
{
    try {
        const ArrayInfo info = getInfo(in, false, true);

        af_dtype inType = info.getType();
        if((inType == c32 || inType == c64)
            && (type == f32 || type == f64)) {
            AF_ERROR("Casting is not allowed from complex (c32/c64) to real (f32/f64) types.\n"
                     "Use abs, real, imag etc to convert complex to floating type.",
                     AF_ERR_TYPE);
        }

        dim4 idims = info.dims();
        if(idims.elements() == 0) {
            dim_t my_dims[] = {0, 0, 0, 0};
            return af_create_handle(out, AF_MAX_DIMS, my_dims, type);
        }

        af_array res = 0;
        if(info.isSparse()) {
            res = castSparse(in, type);
        } else {
            res = cast(in, type);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_cplx(af_array *out, const af_array in, const af_dtype type)
{
    try {
        af_array res;
        ArrayInfo in_info = getInfo(in);

        if (in_info.isDouble()) {
            res = cast(in, c64);
        } else {
            res = cast(in, c32);
        }

        std::swap(*out, res);
    }
    CATCHALL;

    return AF_SUCCESS;
}
