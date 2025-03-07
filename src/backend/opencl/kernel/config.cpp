/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "config.hpp"
namespace arrayfire {
namespace opencl {
namespace kernel {

std::ostream& operator<<(std::ostream& out, const cfloat& var) {
    out << "{" << var.s[0] << "," << var.s[1] << "}";
    return out;
}

std::ostream& operator<<(std::ostream& out, const cdouble& var) {
    out << "{" << var.s[0] << "," << var.s[1] << "}";
    return out;
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
