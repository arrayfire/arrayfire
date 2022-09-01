/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <backend.hpp>
#include <ostream>

namespace oneapi {
static std::ostream& operator<<(std::ostream& out, const cfloat& var) {
    out << "(" << std::real(var) << "," << std::imag(var) << ")";
    return out;
}

static std::ostream& operator<<(std::ostream& out, const cdouble& var) {
    out << "(" << std::real(var) << "," << std::imag(var) << ")";
    return out;
}
}  // namespace oneapi
