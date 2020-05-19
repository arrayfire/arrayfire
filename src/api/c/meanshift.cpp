/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/err_common.hpp>
#include <handle.hpp>
#include <meanshift.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/image.h>

using af::dim4;
using detail::intl;
using detail::meanshift;
using detail::uchar;
using detail::uint;
using detail::uintl;
using detail::ushort;

template<typename T>
static inline af_array mean_shift(const af_array &in, const float &s_sigma,
                                  const float &c_sigma, const unsigned niters,
                                  const bool is_color) {
    return getHandle(
        meanshift<T>(getArray<T>(in), s_sigma, c_sigma, niters, is_color));
}

af_err af_mean_shift(af_array *out, const af_array in,
                     const float spatial_sigma, const float chromatic_sigma,
                     const unsigned num_iterations, const bool is_color) {
    try {
        ARG_ASSERT(2, (spatial_sigma >= 0));
        ARG_ASSERT(3, (chromatic_sigma >= 0));
        ARG_ASSERT(4, (num_iterations > 0));

        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();
        af::dim4 dims         = info.dims();

        DIM_ASSERT(1, (dims.ndims() >= 2));
        if (is_color) { DIM_ASSERT(1, (dims[2] == 3)); }

        af_array output;
        switch (type) {
            case f32:
                output = mean_shift<float>(in, spatial_sigma, chromatic_sigma,
                                           num_iterations, is_color);
                break;
            case f64:
                output = mean_shift<double>(in, spatial_sigma, chromatic_sigma,
                                            num_iterations, is_color);
                break;
            case b8:
                output = mean_shift<char>(in, spatial_sigma, chromatic_sigma,
                                          num_iterations, is_color);
                break;
            case s32:
                output = mean_shift<int>(in, spatial_sigma, chromatic_sigma,
                                         num_iterations, is_color);
                break;
            case u32:
                output = mean_shift<uint>(in, spatial_sigma, chromatic_sigma,
                                          num_iterations, is_color);
                break;
            case s16:
                output = mean_shift<short>(in, spatial_sigma, chromatic_sigma,
                                           num_iterations, is_color);
                break;
            case u16:
                output = mean_shift<ushort>(in, spatial_sigma, chromatic_sigma,
                                            num_iterations, is_color);
                break;
            case s64:
                output = mean_shift<intl>(in, spatial_sigma, chromatic_sigma,
                                          num_iterations, is_color);
                break;
            case u64:
                output = mean_shift<uintl>(in, spatial_sigma, chromatic_sigma,
                                           num_iterations, is_color);
                break;
            case u8:
                output = mean_shift<uchar>(in, spatial_sigma, chromatic_sigma,
                                           num_iterations, is_color);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
