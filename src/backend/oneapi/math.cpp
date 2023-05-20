/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "math.hpp"
#include <common/half.hpp>

namespace arrayfire {
namespace oneapi {

cfloat division(cfloat lhs, double rhs) {
    cfloat retVal(real(lhs) / rhs, imag(lhs) / rhs);
    return retVal;
}

cdouble division(cdouble lhs, double rhs) {
    cdouble retVal(real(lhs) / rhs, imag(lhs) / rhs);
    return retVal;
}
}  // namespace oneapi
}  // namespace arrayfire
