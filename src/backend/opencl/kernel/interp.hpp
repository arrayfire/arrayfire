/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/TemplateArg.hpp>
#include <af/defines.h>

#include <array>
#include <string>

namespace arrayfire {
namespace opencl {
namespace kernel {

static void addInterpEnumOptions(std::vector<std::string>& options) {
    static std::array<std::string, 10> enOpts = {
        DefineKeyValue(AF_INTERP_NEAREST, static_cast<int>(AF_INTERP_NEAREST)),
        DefineKeyValue(AF_INTERP_LINEAR, static_cast<int>(AF_INTERP_LINEAR)),
        DefineKeyValue(AF_INTERP_BILINEAR,
                       static_cast<int>(AF_INTERP_BILINEAR)),
        DefineKeyValue(AF_INTERP_CUBIC, static_cast<int>(AF_INTERP_CUBIC)),
        DefineKeyValue(AF_INTERP_LOWER, static_cast<int>(AF_INTERP_LOWER)),
        DefineKeyValue(AF_INTERP_LINEAR_COSINE,
                       static_cast<int>(AF_INTERP_LINEAR_COSINE)),
        DefineKeyValue(AF_INTERP_BILINEAR_COSINE,
                       static_cast<int>(AF_INTERP_BILINEAR_COSINE)),
        DefineKeyValue(AF_INTERP_BICUBIC, static_cast<int>(AF_INTERP_BICUBIC)),
        DefineKeyValue(AF_INTERP_CUBIC_SPLINE,
                       static_cast<int>(AF_INTERP_CUBIC_SPLINE)),
        DefineKeyValue(AF_INTERP_BICUBIC_SPLINE,
                       static_cast<int>(AF_INTERP_BICUBIC_SPLINE)),
    };
    options.insert(std::end(options), std::begin(enOpts), std::end(enOpts));
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
