/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/defines.h>
#include <string>

#define ADD_ENUM_OPTION(options, name) \
    do { options << " -D " #name "=" << name; } while (0)

namespace opencl {
namespace kernel {

static void addInterpEnumOptions(std::ostringstream &options) {
    ADD_ENUM_OPTION(options, AF_INTERP_NEAREST);
    ADD_ENUM_OPTION(options, AF_INTERP_LINEAR);
    ADD_ENUM_OPTION(options, AF_INTERP_BILINEAR);
    ADD_ENUM_OPTION(options, AF_INTERP_CUBIC);
    ADD_ENUM_OPTION(options, AF_INTERP_LOWER);
    ADD_ENUM_OPTION(options, AF_INTERP_LINEAR_COSINE);
    ADD_ENUM_OPTION(options, AF_INTERP_BILINEAR_COSINE);
    ADD_ENUM_OPTION(options, AF_INTERP_BICUBIC);
    ADD_ENUM_OPTION(options, AF_INTERP_CUBIC_SPLINE);
    ADD_ENUM_OPTION(options, AF_INTERP_BICUBIC_SPLINE);
}
}  // namespace kernel
}  // namespace opencl
