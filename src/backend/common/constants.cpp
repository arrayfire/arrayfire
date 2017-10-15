/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <limits>
#include <af/constants.h>

namespace af
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const double Inf = std::numeric_limits<double>::infinity();
    const double Pi  = 3.1415926535897932384626433832795028841971693993751;
}
