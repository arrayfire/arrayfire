/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <bilateral.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>

#include <type_traits>

using af::dim4;
using detail::bilateral;
using detail::uchar;
using detail::uint;
using detail::ushort;
using std::conditional;
using std::is_same;

template<typename T>
inline af_array bilateral(const af_array &in, const float &sp_sig,
                          const float &chr_sig) {
    using OutType =
        typename conditional<is_same<T, double>::value, double, float>::type;
    return getHandle(bilateral<T, OutType>(getArray<T>(in), sp_sig, chr_sig));
}

af_err af_bilateral(af_array *out, const af_array in, const float ssigma,
                    const float csigma, const bool iscolor) {
    UNUSED(iscolor);
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();
        af::dim4 dims         = info.dims();

        DIM_ASSERT(1, (dims.ndims() >= 2));

        af_array output = nullptr;
        switch (type) {
            case f64: output = bilateral<double>(in, ssigma, csigma); break;
            case f32: output = bilateral<float>(in, ssigma, csigma); break;
            case b8: output = bilateral<char>(in, ssigma, csigma); break;
            case s32: output = bilateral<int>(in, ssigma, csigma); break;
            case u32: output = bilateral<uint>(in, ssigma, csigma); break;
            case u8: output = bilateral<uchar>(in, ssigma, csigma); break;
            case s16: output = bilateral<short>(in, ssigma, csigma); break;
            case u16: output = bilateral<ushort>(in, ssigma, csigma); break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
