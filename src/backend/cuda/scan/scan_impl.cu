/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// This file instantiates scan.cu as separate object files from CMake
// The line below is read by CMake to determenine the instantiations
// SCAN_BINARY_OPS:af_add_t af_mul_t af_max_t af_min_t
#include <kernel/scan.hpp>

namespace cuda {

#define INSTANTIATE_SCAN(ROp, Ti, To)\
    template Array<To> scan<ROp, Ti, To>(const Array<Ti> &in, const int dim, bool inclusive_scan);

#define INSTANTIATE_SCAN_ALL(ROp)                       \
    INSTANTIATE_SCAN(ROp, float  , float  )             \
    INSTANTIATE_SCAN(ROp, double , double )             \
    INSTANTIATE_SCAN(ROp, cfloat , cfloat )             \
    INSTANTIATE_SCAN(ROp, cdouble, cdouble)             \
    INSTANTIATE_SCAN(ROp, int    , int    )             \
    INSTANTIATE_SCAN(ROp, uint   , uint   )             \
    INSTANTIATE_SCAN(ROp, intl   , intl   )             \
    INSTANTIATE_SCAN(ROp, uintl  , uintl  )             \
    INSTANTIATE_SCAN(ROp, char   , int    )             \
    INSTANTIATE_SCAN(ROp, char   , uint   )             \
    INSTANTIATE_SCAN(ROp, uchar  , uint   )             \
    INSTANTIATE_SCAN(ROp, short  , int    )             \
    INSTANTIATE_SCAN(ROp, ushort , uint   )

INSTANTIATE_SCAN_ALL(SCAN_BINARY_OP)

}