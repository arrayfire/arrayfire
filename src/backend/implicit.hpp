/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/array.h>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <optypes.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <types.hpp>
#include <cast.hpp>

using namespace detail;

af_dtype implicit(const af_array lhs, const af_array rhs);
af_dtype implicit(const af_dtype lty, const af_dtype rty);

template<typename To> af_array cast(const af_array in)
{
    const ArrayInfo info = getInfo(in);
    switch (info.getType()) {
    case f32: return getHandle(*cast<To, float  >(getArray<float  >(in)));
    case f64: return getHandle(*cast<To, double >(getArray<double >(in)));
    case c32: return getHandle(*cast<To, cfloat >(getArray<cfloat >(in)));
    case c64: return getHandle(*cast<To, cdouble>(getArray<cdouble>(in)));
    case s32: return getHandle(*cast<To, int    >(getArray<int    >(in)));
    case u32: return getHandle(*cast<To, uint   >(getArray<uint   >(in)));
    case s8 : return getHandle(*cast<To, char   >(getArray<char   >(in)));
    case u8 : return getHandle(*cast<To, uchar  >(getArray<uchar  >(in)));
    case b8 : return getHandle(*cast<To, uchar  >(getArray<uchar  >(in)));
    default: TYPE_ERROR(1, info.getType());
    }
}

af_array cast(const af_array in, const af_dtype type);
