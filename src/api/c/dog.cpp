/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <arith.hpp>
#include <backend.hpp>
#include <common/err_common.hpp>
#include <convolve.hpp>
#include <handle.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>
#include <af/vision.h>

using af::dim4;
using detail::arithOp;
using detail::Array;
using detail::convolve;
using detail::uchar;
using detail::uint;
using detail::ushort;

template<typename T, typename accT>
static af_array dog(const af_array& in, const int radius1, const int radius2) {
    af_array g1, g2;
    g1 = g2 = 0;
    AF_CHECK(
        af_gaussian_kernel(&g1, 2 * radius1 + 1, 2 * radius1 + 1, 0.0, 0.0));
    AF_CHECK(
        af_gaussian_kernel(&g2, 2 * radius2 + 1, 2 * radius2 + 1, 0.0, 0.0));

    Array<accT> input = castArray<accT>(in);
    dim4 iDims        = input.dims();

    AF_BATCH_KIND bkind = iDims[2] > 1 ? AF_BATCH_LHS : AF_BATCH_NONE;

    Array<accT> smth1 =
        convolve<accT, accT>(input, castArray<accT>(g1), bkind, 2, false);
    Array<accT> smth2 =
        convolve<accT, accT>(input, castArray<accT>(g2), bkind, 2, false);
    Array<accT> retVal = arithOp<accT, af_sub_t>(smth1, smth2, iDims);

    AF_CHECK(af_release_array(g1));
    AF_CHECK(af_release_array(g2));

    return getHandle<accT>(retVal);
}

af_err af_dog(af_array* out, const af_array in, const int radius1,
              const int radius2) {
    try {
        const ArrayInfo& info = getInfo(in);
        dim4 inDims           = info.dims();
        ARG_ASSERT(1, (inDims.ndims() >= 2));
        ARG_ASSERT(1, (inDims.ndims() <= 3));

        af_array output;
        af_dtype type = info.getType();
        switch (type) {
            case f32: output = dog<float, float>(in, radius1, radius2); break;
            case f64: output = dog<double, double>(in, radius1, radius2); break;
            case b8: output = dog<char, float>(in, radius1, radius2); break;
            case s32: output = dog<int, float>(in, radius1, radius2); break;
            case u32: output = dog<uint, float>(in, radius1, radius2); break;
            case s16: output = dog<short, float>(in, radius1, radius2); break;
            case u16: output = dog<ushort, float>(in, radius1, radius2); break;
            case u8: output = dog<uchar, float>(in, radius1, radius2); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
